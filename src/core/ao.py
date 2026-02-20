"""Activation-oracle utilities used by this repo.

This is the project-local AO runtime wrapper used by training/eval scripts.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._dynamo as dynamo
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cot_utils import (
    find_sentence_boundary_positions,
    layer_percent_to_layer,
    split_cot_into_sentences,
)


AO_CHECKPOINTS = {
    "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
    "Qwen/Qwen3-4B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B",
    "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
}

SPECIAL_TOKEN = " ?"


class EarlyStopException(Exception):
    pass


def _is_blackwell_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major_caps = [torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())]
    except Exception:
        return False
    return bool(major_caps) and max(major_caps) >= 12


def choose_attn_implementation(model_name: str) -> str:
    """Choose attention backend with safe defaults for Blackwell GPUs."""
    if os.environ.get("COT_ORACLE_FORCE_SDPA") == "1":
        return "sdpa"
    if "gemma" in model_name.lower():
        return "eager"

    blackwell_detected = _is_blackwell_gpu()
    allow_flash2 = os.environ.get("COT_ORACLE_ALLOW_FLASH2") == "1"
    if blackwell_detected and not allow_flash2:
        print("Blackwell GPU detected; using SDPA for compatibility (set COT_ORACLE_ALLOW_FLASH2=1 to override).")
        return "sdpa"

    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model_with_ao(
    model_name: str,
    use_8bit: bool = False,
    device: str = "cuda",
) -> tuple[PeftModel, AutoTokenizer]:
    """Load base model + AO adapter."""
    del device  # device_map=auto handles placement

    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "attn_implementation": choose_attn_implementation(model_name),
    }

    if use_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    # Add a local adapter name so PeftModel APIs are always present.
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    ao_path = AO_CHECKPOINTS[model_name]
    print(f"Loading AO LoRA: {ao_path}")
    sanitized = ao_path.replace(".", "_")
    if sanitized not in model.peft_config:
        model.load_adapter(ao_path, adapter_name=sanitized, is_trainable=False, low_cpu_mem_usage=True)

    return model, tokenizer


def get_hf_submodule(model, layer: int, use_lora: bool = False):
    """Get a transformer block across common HF/PEFT wrapper variants."""
    del use_lora  # retained for compatibility with older call sites

    paths_to_try = [
        lambda: model.base_model.model.model.layers[layer],
        lambda: model.base_model.model.layers[layer],
        lambda: model.base_model.language_model.layers[layer],
        lambda: model.model.model.layers[layer],
        lambda: model.model.layers[layer],
        lambda: model.language_model.layers[layer],
    ]

    for path_fn in paths_to_try:
        try:
            return path_fn()
        except (AttributeError, IndexError):
            continue

    raise ValueError(f"Could not find layer {layer} in model {model.config._name_or_path}")


def collect_activations(
    model,
    layer: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Collect activations from one layer. Returns [B, L, D]."""
    activations = None
    submodule = get_hf_submodule(model, layer)

    def hook_fn(module, inputs, outputs):
        del module, inputs
        nonlocal activations
        activations = outputs[0] if isinstance(outputs, tuple) else outputs
        raise EarlyStopException()

    handle = submodule.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    except EarlyStopException:
        pass
    finally:
        handle.remove()

    return activations


def collect_activations_at_positions(
    model,
    tokenizer,
    text: str,
    layer: int,
    positions: list[int],
    device: str = "cuda",
) -> torch.Tensor:
    """Extract activations at token positions. Returns [K, D]."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    was_training = model.training
    model.eval()
    with model.disable_adapter():
        acts_BLD = collect_activations(
            model,
            layer,
            inputs["input_ids"],
            inputs["attention_mask"],
        )
    if was_training:
        model.train()

    return acts_BLD[0, positions, :].detach()


@contextlib.contextmanager
def add_hook(module, hook):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_steering_hook(vectors, positions, device, dtype, steering_coefficient=1.0):
    """Norm-matched additive steering hook."""
    normed = torch.nn.functional.normalize(vectors, dim=-1).detach()

    def hook_fn(module, _input, output):
        del module, _input

        if isinstance(output, tuple):
            resid, *rest = output
            is_tuple = True
        else:
            resid = output
            is_tuple = False

        _b, l, _d = resid.shape
        if l <= 1:
            return (resid, *rest) if is_tuple else resid

        pos = torch.tensor(positions, dtype=torch.long, device=device)
        orig = resid[0, pos, :]
        norms = orig.norm(dim=-1, keepdim=True)
        steered = (normed.to(device, dtype) * norms * steering_coefficient).detach()
        resid[0, pos, :] = steered + orig

        return (resid, *rest) if is_tuple else resid

    return hook_fn


@dynamo.disable
@torch.no_grad()
def run_oracle_on_activations(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    activations: torch.Tensor,
    oracle_prompt: str,
    model_name: str,
    injection_layer: int = 1,
    act_layer: int | None = None,
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> str:
    """Run the oracle adapter on provided activation vectors."""
    dtype = torch.bfloat16
    num_positions = activations.shape[0]

    if act_layer is None:
        act_layer = layer_percent_to_layer(model_name, 50)

    prefix = f"Layer: {act_layer}\n" + SPECIAL_TOKEN * num_positions + " \n"
    full_prompt = prefix + oracle_prompt

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    special_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    assert len(special_id) == 1, f"Expected single token for '{SPECIAL_TOKEN}', got {len(special_id)}"
    special_id = special_id[0]

    positions = []
    for i, tid in enumerate(input_ids):
        if tid == special_id and len(positions) < num_positions:
            positions.append(i)
    assert len(positions) == num_positions, (
        f"Found {len(positions)} placeholder positions, expected {num_positions}"
    )

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    ao_path = AO_CHECKPOINTS[model_name]
    sanitized = ao_path.replace(".", "_")
    try:
        model.set_adapter(sanitized)
    except ValueError:
        try:
            model.set_adapter("default")
        except ValueError as exc:
            if hasattr(model, "peft_config"):
                available = list(model.peft_config.keys())
                raise RuntimeError(f"Cannot set adapter. Available: {available}") from exc
            raise

    hook_fn = get_steering_hook(
        vectors=activations,
        positions=positions,
        device=device,
        dtype=dtype,
    )

    injection_submodule = get_hf_submodule(model, injection_layer, use_lora=True)

    with add_hook(injection_submodule, hook_fn):
        output = model.generate(
            input_ids=input_tensor,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(output[0][len(input_ids) :], skip_special_tokens=True)


def generate_cot(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 1024,
    device: str = "cuda",
) -> str:
    """Generate CoT text from base model (adapter disabled)."""
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with model.disable_adapter():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)


def batch_generate_cot(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    questions: list[str],
    max_new_tokens: int = 1024,
    device: str = "cuda",
    batch_size: int = 8,
    enable_thinking: bool = True,
) -> list[str]:
    """Batch CoT generation."""
    all_responses: list[str] = []

    with model.disable_adapter():
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            formatted = []
            for question in batch_questions:
                messages = [{"role": "user", "content": question}]
                formatted.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking,
                    )
                )

            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            prompt_lens = [inputs["attention_mask"][j].sum().item() for j in range(len(batch_questions))]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            for j, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[prompt_lens[j] :],
                    skip_special_tokens=not enable_thinking,
                )
                all_responses.append(response)

    return all_responses


def generate_direct_answer(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> str:
    """Generate answer with thinking disabled."""
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with model.disable_adapter():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
