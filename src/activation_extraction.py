"""
Extract activations from model during CoT generation.

Key functionality:
- Hook into model to capture hidden states at each layer
- Extract activations at sentence boundaries during generation
- Support for both "with CoT" and "without CoT" conditions
"""

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


@dataclass
class ActivationCapture:
    """Captured activations from a single forward pass or generation."""
    hidden_states: dict[int, Tensor]  # layer -> [batch, seq, d_model]
    attention_patterns: dict[int, Tensor] | None  # layer -> [batch, heads, seq, seq]
    token_ids: Tensor
    text: str


@dataclass
class TrajectoryActivations:
    """Activations at sentence boundaries throughout a CoT."""
    sentence_activations: list[Tensor]  # [num_sentences] of [d_model]
    sentence_texts: list[str]
    final_activation: Tensor  # At answer token
    full_response: str
    layer: int  # Which layer these came from


class ActivationExtractor:
    """Extract activations from a model during generation."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layers_to_capture: list[int] | None = None,
        capture_attention: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.capture_attention = capture_attention

        # Default to middle layers if not specified
        if layers_to_capture is None:
            n_layers = model.config.num_hidden_layers
            self.layers_to_capture = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
        else:
            self.layers_to_capture = layers_to_capture

        self._hooks = []
        self._captured = {}

    def _get_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._captured[f"layer_{layer_idx}"] = hidden.detach().cpu()
        return hook

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        self._clear_hooks()
        for layer_idx in self.layers_to_capture:
            # This works for most HF models (Llama, Qwen, etc.)
            # Adjust path for different architectures
            try:
                layer = self.model.model.layers[layer_idx]
            except AttributeError:
                # Try alternative paths
                try:
                    layer = self.model.transformer.h[layer_idx]
                except AttributeError:
                    raise ValueError(f"Could not find layer {layer_idx} in model architecture")

            hook = layer.register_forward_hook(self._get_hook(layer_idx))
            self._hooks.append(hook)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._captured = {}

    def get_final_token_activation(
        self,
        prompt: str,
        layer: int | None = None,
    ) -> Tensor:
        """Get activation at the final token position for a prompt.

        Used for the "no CoT" condition - just prompt -> answer.
        """
        if layer is None:
            layer = self.layers_to_capture[len(self.layers_to_capture) // 2]

        self._register_hooks()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model(**inputs)

            # Get activation at last token
            hidden = self._captured[f"layer_{layer}"]
            return hidden[0, -1, :]  # [d_model]
        finally:
            self._clear_hooks()

    def generate_with_activations(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        layer: int | None = None,
        temperature: float = 0.7,
    ) -> tuple[str, Tensor]:
        """Generate response and capture activation at final token.

        Used for the "with CoT" condition.
        Returns (generated_text, final_activation).
        """
        if layer is None:
            layer = self.layers_to_capture[len(self.layers_to_capture) // 2]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Now get activation at the final generated token
        self._register_hooks()
        try:
            with torch.no_grad():
                self.model(generated_ids.unsqueeze(0))
            hidden = self._captured[f"layer_{layer}"]
            final_activation = hidden[0, -1, :]
        finally:
            self._clear_hooks()

        # Strip prompt from generated text
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        response_text = generated_text[len(prompt_text):].strip()

        return response_text, final_activation

    def extract_trajectory(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        layer: int | None = None,
        temperature: float = 0.7,
    ) -> TrajectoryActivations:
        """Generate CoT and extract activations at sentence boundaries.

        This is the key method for the trajectory oracle approach.
        """
        if layer is None:
            layer = self.layers_to_capture[len(self.layers_to_capture) // 2]

        # First generate the full response
        response, final_activation = self.generate_with_activations(
            prompt, max_new_tokens, layer, temperature
        )

        # Split into sentences
        sentences = self._split_into_sentences(response)

        # Now we need activations at the end of each sentence
        # We'll do this by running forward passes on progressively longer prefixes
        sentence_activations = []
        current_text = prompt

        self._register_hooks()
        try:
            for sent in sentences:
                current_text += " " + sent
                inputs = self.tokenizer(current_text, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    self.model(**inputs)
                hidden = self._captured[f"layer_{layer}"]
                sentence_activations.append(hidden[0, -1, :].clone())
        finally:
            self._clear_hooks()

        return TrajectoryActivations(
            sentence_activations=sentence_activations,
            sentence_texts=sentences,
            final_activation=final_activation,
            full_response=response,
            layer=layer,
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def compute_contrastive_vector(
        self,
        prompt_without_cot: str,
        prompt_with_cot: str,
        layer: int | None = None,
        n_cot_samples: int = 1,
        max_new_tokens: int = 512,
    ) -> tuple[Tensor, list[str]]:
        """Compute Y - X contrastive vector.

        X = activation at answer token without CoT
        Y = activation at answer token after CoT (optionally averaged over samples)

        Returns (contrastive_vector, list_of_cot_responses)
        """
        if layer is None:
            layer = self.layers_to_capture[len(self.layers_to_capture) // 2]

        # Get X (no CoT)
        x = self.get_final_token_activation(prompt_without_cot, layer)

        # Get Y (with CoT, possibly multiple samples)
        y_samples = []
        responses = []
        for _ in range(n_cot_samples):
            response, y = self.generate_with_activations(
                prompt_with_cot, max_new_tokens, layer
            )
            y_samples.append(y)
            responses.append(response)

        # Average Y samples
        y_mean = torch.stack(y_samples).mean(dim=0)

        # Contrastive vector
        contrastive = y_mean - x

        return contrastive, responses


def load_model_and_extractor(
    model_name: str = "Qwen/Qwen3-8B",  # Default to Qwen3-8B
    device: str = "cuda",
    layers: list[int] | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ActivationExtractor]:
    """Convenience function to load model and create extractor.

    For Qwen3-8B (36 layers), default extraction is at 50% depth = layer 18.
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Default to 50% depth for extraction (per AO paper)
    if layers is None:
        n_layers = model.config.num_hidden_layers
        layers = [n_layers // 2]  # 50% depth

    extractor = ActivationExtractor(model, tokenizer, layers_to_capture=layers)

    return model, tokenizer, extractor


if __name__ == "__main__":
    # Quick test
    model, tokenizer, extractor = load_model_and_extractor(
        "Qwen/Qwen3-8B",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    prompt = "What is 17 * 24? Think step by step."

    print("Extracting trajectory...")
    traj = extractor.extract_trajectory(prompt, max_new_tokens=200)

    print(f"\nGenerated {len(traj.sentence_texts)} sentences:")
    for i, (sent, act) in enumerate(zip(traj.sentence_texts, traj.sentence_activations)):
        print(f"  {i}: {sent[:60]}... (act norm: {act.norm():.2f})")

    print(f"\nFinal activation norm: {traj.final_activation.norm():.2f}")
