"""
Minimal Activation Oracle library inlined from the AO demo notebook.
This avoids import/path issues with the full AO repo.

Core functions:
- load_model_with_ao: Load base model + AO LoRA
- collect_activations_at_positions: Extract activations from specific token positions
- run_oracle_on_activations: Feed activations to oracle and get response
"""

import os
import re
import contextlib
from dataclasses import dataclass

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._dynamo as dynamo
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from position_encoding import apply_position_encoding

# ============================================================
# Constants
# ============================================================

LAYER_COUNTS = {
    "Qwen/Qwen3-1.7B": 28,
    "Qwen/Qwen3-4B": 36,
    "Qwen/Qwen3-8B": 36,
    "google/gemma-3-1b-it": 26,
}

AO_CHECKPOINTS = {
    "Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
    "Qwen/Qwen3-4B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B",
    "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
}

SPECIAL_TOKEN = " ?"


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    max_layers = LAYER_COUNTS[model_name]
    return int(max_layers * (layer_percent / 100))


# ============================================================
# Model Loading
# ============================================================

def load_model_with_ao(
    model_name: str,
    use_8bit: bool = False,  # NEVER use 8-bit on H100 — bf16 is faster and we have 80GB
    device: str = "cuda",
) -> tuple[PeftModel, AutoTokenizer]:
    """Load base model with AO LoRA adapter. Always bf16 on H100."""
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    # Use flash_attention_2 if available, otherwise sdpa
    try:
        import flash_attn  # noqa: F401
        if "Qwen" in model_name:
            kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        kwargs["attn_implementation"] = "sdpa"
    if use_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    # Add dummy adapter so PeftModel API works
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    # Load AO adapter
    ao_path = AO_CHECKPOINTS[model_name]
    print(f"Loading AO LoRA: {ao_path}")
    sanitized = ao_path.replace(".", "_")
    if sanitized not in model.peft_config:
        model.load_adapter(ao_path, adapter_name=sanitized, is_trainable=False, low_cpu_mem_usage=True)

    return model, tokenizer


# ============================================================
# Activation Collection
# ============================================================

class EarlyStopException(Exception):
    pass


def get_hf_submodule(model, layer: int, use_lora: bool = False):
    """Get a transformer layer module, handling various PEFT wrapping styles."""
    # Try all paths — PeftModel wrapping is always present even with adapters disabled
    paths_to_try = [
        lambda: model.base_model.model.model.layers[layer],  # PeftModel wrapping (deepest)
        lambda: model.base_model.model.layers[layer],        # add_adapter() style
        lambda: model.base_model.language_model.layers[layer],  # gemma-3 with peft
        lambda: model.model.model.layers[layer],              # PeftModel.model = LoraModel
        lambda: model.model.layers[layer],                    # Standard Qwen/Llama (no peft)
        lambda: model.language_model.layers[layer],           # gemma-3 (no peft)
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
    """Collect activations from a specific layer. Returns [batch, seq_len, d_model]."""
    activations = None
    submodule = get_hf_submodule(model, layer)

    def hook_fn(module, inputs, outputs):
        nonlocal activations
        if isinstance(outputs, tuple):
            activations = outputs[0]
        else:
            activations = outputs
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
    """Extract activations at specific token positions. Returns [num_positions, d_model]."""
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

    # Disable LoRA for activation collection (read base model)
    # Use peft's context manager which properly restores adapter state
    was_training = model.training
    model.eval()
    with model.disable_adapter():
        acts_BLD = collect_activations(
            model, layer,
            inputs["input_ids"],
            inputs["attention_mask"],
        )
    if was_training:
        model.train()

    # Index specific positions
    acts = acts_BLD[0, positions, :].detach()  # [num_positions, d_model]
    total_length = inputs["input_ids"].shape[1]
    acts = apply_position_encoding(acts, positions, total_length)
    return acts


# ============================================================
# Steering Hook
# ============================================================

@contextlib.contextmanager
def add_hook(module, hook):
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_steering_hook(vectors, positions, device, dtype, steering_coefficient=1.0):
    """Create norm-matched addition hook. vectors: [K, d_model], positions: list of K ints."""
    normed = torch.nn.functional.normalize(vectors, dim=-1).detach()

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid, *rest = output
            is_tuple = True
        else:
            resid = output
            is_tuple = False

        B, L, D = resid.shape
        if L <= 1:
            return (resid, *rest) if is_tuple else resid

        pos = torch.tensor(positions, dtype=torch.long, device=device)
        orig = resid[0, pos, :]  # [K, D]
        norms = orig.norm(dim=-1, keepdim=True)  # [K, 1]
        steered = (normed.to(device, dtype) * norms * steering_coefficient).detach()
        resid[0, pos, :] = steered + orig

        return (resid, *rest) if is_tuple else resid

    return hook_fn


# ============================================================
# Oracle Inference
# ============================================================

@dynamo.disable
@torch.no_grad()
def run_oracle_on_activations(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    activations: torch.Tensor,  # [num_positions, d_model]
    oracle_prompt: str,
    model_name: str,
    injection_layer: int = 1,
    act_layer: int | None = None,
    max_new_tokens: int = 100,
    device: str = "cuda",
) -> str:
    """
    Feed activations to the oracle and get a response.

    Args:
        activations: [num_positions, d_model] tensor of collected activations
        oracle_prompt: question to ask the oracle
        act_layer: layer the activations came from (for prompt formatting)
    """
    dtype = torch.bfloat16
    num_positions = activations.shape[0]

    if act_layer is None:
        act_layer = layer_percent_to_layer(model_name, 50)

    # Build prompt with placeholder tokens
    prefix = f"Layer: {act_layer}\n" + SPECIAL_TOKEN * num_positions + " \n"
    full_prompt = prefix + oracle_prompt

    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(formatted, add_special_tokens=False)

    # Find placeholder positions
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

    # Set oracle adapter active
    # After disable_adapter context managers, adapters are already re-enabled
    ao_path = AO_CHECKPOINTS[model_name]
    sanitized = ao_path.replace(".", "_")
    try:
        model.set_adapter(sanitized)
    except ValueError:
        # During training, adapter is named "default" instead of the checkpoint name
        try:
            model.set_adapter("default")
        except ValueError:
            # List what adapters exist for debugging
            if hasattr(model, 'peft_config'):
                available = list(model.peft_config.keys())
                raise RuntimeError(f"Cannot set adapter. Available: {available}")
            raise

    # Create steering hook
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

    response = tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)
    return response


# ============================================================
# CoT Utilities
# ============================================================

def generate_cot(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 1024,
    device: str = "cuda",
) -> str:
    """Generate a CoT response with <think> tags."""
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with model.disable_adapter():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return response


def batch_generate_cot(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    questions: list[str],
    max_new_tokens: int = 1024,
    device: str = "cuda",
    batch_size: int = 8,
    enable_thinking: bool = True,
) -> list[str]:
    """Batch generate CoT responses. Much faster than one-at-a-time on H100."""
    all_responses = []

    with model.disable_adapter():
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            formatted = []
            for q in batch_questions:
                messages = [{"role": "user", "content": q}]
                formatted.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ))

            inputs = tokenizer(
                formatted, return_tensors="pt", padding=True, truncation=True,
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
                    output[prompt_lens[j]:],
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
    """Generate a direct answer without CoT (thinking disabled)."""
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with model.disable_adapter():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def split_cot_into_sentences(cot_text: str) -> list[str]:
    """Split CoT text into sentences. Removes <think> tags first."""
    # Remove think tags
    text = re.sub(r'<think>|</think>', '', cot_text).strip()
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter empty
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def find_sentence_boundary_positions(
    tokenizer: AutoTokenizer,
    formatted_text: str,
    sentences: list[str],
) -> list[int]:
    """
    Find the token positions of the last token of each sentence in the formatted text.
    Returns a list of token positions (indices into the tokenized sequence).
    """
    tokens = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=False)
    token_ids = tokens["input_ids"][0].tolist()
    full_text_decoded = tokenizer.decode(token_ids)

    positions = []
    search_start = 0

    for sentence in sentences:
        # Find end of this sentence in full decoded text
        # Use last few chars of sentence as anchor
        anchor = sentence[-20:] if len(sentence) > 20 else sentence
        idx = full_text_decoded.find(anchor, search_start)
        if idx == -1:
            # Try with shorter anchor
            anchor = sentence[-10:] if len(sentence) > 10 else sentence
            idx = full_text_decoded.find(anchor, search_start)
        if idx == -1:
            continue

        char_end = idx + len(anchor)
        search_start = char_end

        # Convert character position to token position
        # Decode tokens one by one to find which token covers char_end
        char_count = 0
        token_pos = -1
        for t_idx, t_id in enumerate(token_ids):
            decoded = tokenizer.decode([t_id])
            char_count += len(decoded)
            if char_count >= char_end:
                token_pos = t_idx
                break

        if token_pos >= 0:
            positions.append(token_pos)

    return positions


@dataclass
class SignsOfLifeResult:
    question: str
    cot_text: str
    sentences: list[str]
    boundary_positions: list[int]
    oracle_responses: dict[str, str]  # prompt -> response


# ============================================================
# Math Problems for Testing
# ============================================================

MATH_PROBLEMS = [
    "What is 17 * 24?",
    "If a train travels at 60 mph for 2.5 hours, how far does it go?",
    "What is the sum of the first 10 positive integers?",
    "A rectangle has a length of 12 cm and width of 8 cm. What is its area?",
    "If 3x + 7 = 22, what is x?",
    "What is 15% of 200?",
    "How many ways can you arrange the letters in the word MATH?",
    "What is the greatest common divisor of 48 and 36?",
    "If a pizza is cut into 8 equal slices and you eat 3, what fraction remains?",
    "What is 2^10?",
    "A bag has 3 red, 4 blue, and 5 green marbles. What is the probability of drawing a red marble?",
    "What is the perimeter of a square with side length 7?",
    "If f(x) = 2x + 3, what is f(5)?",
    "What is 144 divided by 12?",
    "How many prime numbers are less than 20?",
    "What is the average of 15, 20, 25, 30, and 35?",
    "A car depreciates by 15% each year. If it costs $20000, what is it worth after 1 year?",
    "What is the volume of a cube with side length 5?",
    "If you flip a coin 3 times, how many possible outcomes are there?",
    "What is the square root of 169?",
]
