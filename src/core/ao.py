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
TRAINED_PLACEHOLDER = " ¶"


class EarlyStopException(Exception):
    pass


def _active_adapter_name(model: PeftModel) -> str | None:
    """Return active adapter name when available."""
    active = getattr(model, "active_adapter", None)
    if isinstance(active, str):
        return active
    if isinstance(active, (list, tuple)) and active:
        if isinstance(active[0], str):
            return active[0]
    return None


@contextlib.contextmanager
def using_adapter(model: PeftModel, adapter_name: str | None):
    """Temporarily run with a specific adapter, or with adapters disabled."""
    previous_adapter = _active_adapter_name(model)
    if adapter_name is None:
        # peft API varies: disable_adapter() (context mgr) vs disable_adapters()
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                yield
        elif hasattr(model, "disable_adapters"):
            model.disable_adapters()
            try:
                yield
            finally:
                model.enable_adapters()
        else:
            yield  # no adapter machinery — just run as-is
        return

    if adapter_name not in getattr(model, "peft_config", {}):
        available = list(getattr(model, "peft_config", {}).keys())
        raise ValueError(f"Adapter '{adapter_name}' is not loaded. Available adapters: {available}")

    model.set_adapter(adapter_name)
    try:
        yield
    finally:
        if previous_adapter and previous_adapter in getattr(model, "peft_config", {}):
            model.set_adapter(previous_adapter)


def load_extra_adapter(
    model: PeftModel,
    adapter_path: str,
    adapter_name: str = "generator",
) -> str:
    """Load an extra inference-only LoRA adapter and return its adapter name."""
    if adapter_name in model.peft_config:
        return adapter_name
    print(f"Loading extra LoRA adapter '{adapter_name}' from: {adapter_path}")
    model.load_adapter(
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=False,
        low_cpu_mem_usage=True,
    )
    return adapter_name


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
    adapter_name: str | None = None,
    position_encoding: bool = False,
    pe_alpha: float = 0.1,
) -> torch.Tensor:
    """Extract activations at token positions. Returns [K, D]."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    was_training = model.training
    model.eval()
    with using_adapter(model, adapter_name):
        acts_BLD = collect_activations(
            model,
            layer,
            inputs["input_ids"],
            inputs["attention_mask"],
        )
    if was_training:
        model.train()

    acts = acts_BLD[0, positions, :].detach()
    if position_encoding:
        from position_encoding import apply_position_encoding
        total_length = inputs["input_ids"].shape[1]
        acts = apply_position_encoding(acts, positions, total_length, alpha=pe_alpha)
    return acts


@contextlib.contextmanager
def add_hook(module, hook):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_steering_hook(vectors, positions, device, dtype, steering_coefficient=1.0):
    """Norm-matched additive steering hook (batch=1 only)."""
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


def get_batched_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[list[int]],
    device: str | torch.device,
    dtype: torch.dtype,
    steering_coefficient: float = 1.0,
):
    """Batched norm-matched additive steering hook.

    Based on Adam Karvonen's get_hf_activation_steering_hook from
    activation_oracles. Supports variable K per batch element (ragged).

    Args:
        vectors: len-B list of [K_b, D] tensors (K_b can differ per item).
        positions: len-B list of position lists (already adjusted for left-padding).
        device: Target device.
        dtype: Target dtype.
        steering_coefficient: Scaling factor for injected vectors.
    """
    assert len(vectors) == len(positions)
    B = len(vectors)
    normed_list = [torch.nn.functional.normalize(v, dim=-1).detach() for v in vectors]

    def hook_fn(module, _input, output):
        del module, _input

        if isinstance(output, tuple):
            resid, *rest = output
            is_tuple = True
        else:
            resid = output
            is_tuple = False

        _b, l, _d = resid.shape
        # Only inject during prompt processing, not autoregressive steps
        if l <= 1:
            return (resid, *rest) if is_tuple else resid

        for b in range(min(_b, B)):
            pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
            orig = resid[b, pos_b, :]
            norms = orig.norm(dim=-1, keepdim=True)
            steered = (normed_list[b].to(device, dtype) * norms * steering_coefficient).detach()
            resid[b, pos_b, :] = steered + orig

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
    placeholder_token: str | None = None,
    oracle_adapter_name: str | None = None,
) -> str:
    """Run an oracle adapter on provided activation vectors.

    Args:
        placeholder_token: Token to use for activation injection positions.
            Default (None) uses SPECIAL_TOKEN (" ?") for original AO.
            Pass TRAINED_PLACEHOLDER (" ¶") for trained oracle.
        oracle_adapter_name: Adapter name to activate for oracle inference.
            Default (None) auto-detects original AO adapter.
            Pass "default" or "trained" for trained oracle during training/eval.
    """
    dtype = torch.bfloat16
    num_positions = activations.shape[0]
    ph_token = placeholder_token or SPECIAL_TOKEN

    if act_layer is None:
        act_layer = layer_percent_to_layer(model_name, 50)

    if isinstance(act_layer, (list, tuple)):
        layers_str = ", ".join(str(l) for l in act_layer)
        prefix = f"Layer: {layers_str}\n" + ph_token * num_positions + " \n"
    else:
        prefix = f"Layer: {act_layer}\n" + ph_token * num_positions + " \n"
    full_prompt = prefix + oracle_prompt

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id) == 1, f"Expected single token for '{ph_token}', got {len(ph_id)}"
    ph_id = ph_id[0]

    positions = []
    for i, tid in enumerate(input_ids):
        if tid == ph_id and len(positions) < num_positions:
            positions.append(i)
    assert len(positions) == num_positions, (
        f"Found {len(positions)} placeholder positions for '{ph_token}', expected {num_positions}"
    )

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    previous_adapter = _active_adapter_name(model)
    active_adapter = oracle_adapter_name
    if oracle_adapter_name is not None:
        model.set_adapter(oracle_adapter_name)
    else:
        # Auto-detect original AO adapter
        ao_path = AO_CHECKPOINTS[model_name]
        active_adapter = ao_path.replace(".", "_")
        try:
            model.set_adapter(active_adapter)
        except ValueError:
            try:
                model.set_adapter("default")
                active_adapter = "default"
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

    was_training = model.training
    model.eval()
    try:
        with add_hook(injection_submodule, hook_fn):
            output = model.generate(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        if was_training:
            model.train()
        if previous_adapter and previous_adapter in getattr(model, "peft_config", {}) and previous_adapter != active_adapter:
            model.set_adapter(previous_adapter)

    return tokenizer.decode(output[0][len(input_ids) :], skip_special_tokens=True)


@dynamo.disable
@torch.no_grad()
def run_oracle_with_answer_logprobs(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    activations: torch.Tensor,
    oracle_prompt: str,
    answer_tokens: list[str],
    model_name: str,
    injection_layer: int = 1,
    act_layer: int | None = None,
    device: str = "cuda",
    placeholder_token: str | None = None,
    oracle_adapter_name: str | None = None,
) -> dict[str, float]:
    """Run oracle forward pass and return logprobs for specified answer tokens.

    Instead of generating text, does a single forward pass with activation
    injection and extracts the logprob distribution over answer_tokens at
    the last position. Used for forced_answer_entropy eval.

    Args:
        answer_tokens: List of token strings (e.g. ["A", "B", "C", "D"]).

    Returns:
        Dict mapping each answer token to its logprob (unnormalized log probability).
    """
    import math

    dtype = torch.bfloat16
    num_positions = activations.shape[0]
    ph_token = placeholder_token or SPECIAL_TOKEN

    if act_layer is None:
        act_layer = layer_percent_to_layer(model_name, 50)

    if isinstance(act_layer, (list, tuple)):
        layers_str = ", ".join(str(l) for l in act_layer)
        prefix = f"Layer: {layers_str}\n" + ph_token * num_positions + " \n"
    else:
        prefix = f"Layer: {act_layer}\n" + ph_token * num_positions + " \n"
    full_prompt = prefix + oracle_prompt

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    ph_id = tokenizer.encode(ph_token, add_special_tokens=False)
    assert len(ph_id) == 1, f"Expected single token for '{ph_token}', got {len(ph_id)}"
    ph_id = ph_id[0]

    positions = []
    for i, tid in enumerate(input_ids):
        if tid == ph_id and len(positions) < num_positions:
            positions.append(i)
    assert len(positions) == num_positions, (
        f"Found {len(positions)} placeholder positions for '{ph_token}', expected {num_positions}"
    )

    # Resolve answer token IDs
    answer_token_ids = {}
    for tok_str in answer_tokens:
        candidates = tokenizer.encode(tok_str, add_special_tokens=False)
        if candidates:
            answer_token_ids[tok_str] = candidates[-1]

    input_tensor = torch.tensor([input_ids], device=device)
    attn_mask = torch.ones_like(input_tensor)

    previous_adapter = _active_adapter_name(model)
    active_adapter = oracle_adapter_name
    if oracle_adapter_name is not None:
        model.set_adapter(oracle_adapter_name)
    else:
        ao_path = AO_CHECKPOINTS[model_name]
        active_adapter = ao_path.replace(".", "_")
        try:
            model.set_adapter(active_adapter)
        except ValueError:
            try:
                model.set_adapter("default")
                active_adapter = "default"
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

    was_training = model.training
    model.eval()
    try:
        with add_hook(injection_submodule, hook_fn):
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attn_mask,
            )
    finally:
        if was_training:
            model.train()
        if previous_adapter and previous_adapter in getattr(model, "peft_config", {}) and previous_adapter != active_adapter:
            model.set_adapter(previous_adapter)

    # Extract logprobs at the last position (next-token prediction)
    logits = outputs.logits[0, -1, :]  # [vocab_size]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    result = {}
    for tok_str, tid in answer_token_ids.items():
        result[tok_str] = log_probs[tid].item()

    return result


def generate_cot(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 1024,
    device: str = "cuda",
    adapter_name: str | None = None,
    temperature: float | None = None,
) -> str:
    """Generate CoT text from base model (adapter disabled).

    Args:
        temperature: If set, use sampling with this temperature. None = greedy.
    """
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False
    was_training = model.training
    model.eval()
    with using_adapter(model, adapter_name):
        output = model.generate(
            **inputs,
            **gen_kwargs,
        )
    if was_training:
        model.train()
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)


def batch_generate_cot(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    questions: list[str],
    max_new_tokens: int = 1024,
    device: str = "cuda",
    batch_size: int = 8,
    enable_thinking: bool = True,
    adapter_name: str | None = None,
) -> list[str]:
    """Batch CoT generation."""
    all_responses: list[str] = []
    was_training = model.training
    model.eval()

    with using_adapter(model, adapter_name):
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

    if was_training:
        model.train()
    return all_responses


def generate_direct_answer(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
    adapter_name: str | None = None,
    temperature: float | None = None,
) -> str:
    """Generate answer with thinking disabled.

    Args:
        temperature: If set, use sampling with this temperature. None = greedy.
    """
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False
    with using_adapter(model, adapter_name):
        output = model.generate(
            **inputs,
            **gen_kwargs,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
