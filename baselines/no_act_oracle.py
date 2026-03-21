"""Baseline: No-activations oracle (text-only, same LoRA finetune trained with --no-activations).

Shows how much value the activations add over just reading the CoT text.

Unified API: accepts test_data + activations from eval_comprehensive.
"""

import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.ao import load_extra_adapter, using_adapter


def run_no_act_oracle(
    test_data: list[dict],
    activations: list[torch.Tensor],  # unused (text-only), kept for signature consistency
    layers: list[int],
    task_def,
    model,
    tokenizer,
    *,
    checkpoint: str = "",
    max_new_tokens: int = 150,
    device: str = "cuda",
) -> list[str]:
    """Run no-activations oracle. Returns list[str] raw predictions."""
    if not checkpoint or not Path(checkpoint).exists():
        print(f"  WARNING: No-act oracle checkpoint not found at {checkpoint}")
        return [""] * len(test_data)

    adapter_name = load_extra_adapter(model, checkpoint, adapter_name="no_act")
    supervisor_context = task_def.supervisor_context

    predictions = []
    model.eval()

    for item in tqdm(test_data, desc="No-act oracle"):
        cot_text = item.get(supervisor_context, "")[:4000]
        question = item.get("question", item.get("prompt", ""))
        prompt_text = f"Question: {question}\nChain of thought: {cot_text}\n\n{item.get('prompt', '')}"

        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        input_ids = tokenizer.encode(formatted, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=device)
        attn_mask = torch.ones_like(input_tensor)

        with using_adapter(model, adapter_name):
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_tensor, attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                )

        response = tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)
        predictions.append(response)

    return predictions
