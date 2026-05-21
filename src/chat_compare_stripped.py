"""
Public demo: CoT Oracle vs original Activation Oracle.

Stripped-down version of chat_compare.py for https://ao.celeste.computer.
Serves a minimal web UI: user types a question -> model generates CoT ->
activations collected at stride positions -> user drag-selects activation
cells -> both the original AO and the trained CoT oracle are queried with
the selected activations and a user-supplied prompt -> outputs side-by-side.

A global FIFO queue (asyncio.Lock + waiting counter) ensures only one
GPU-touching request runs at a time.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / "ao_reference"))

from cot_utils import get_cot_positions, layer_percent_to_layer
from core.ao import (
    AO_CHECKPOINTS,
    SPECIAL_TOKEN,
    add_hook,
    choose_attn_implementation,
    collect_activations_at_positions,
    get_hf_submodule,
    get_steering_hook,
)

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path.home() / ".env")

TRAINED_PLACEHOLDER = " ?"


AUTO_8BIT_MEMORY_THRESHOLD = 30 * 1024 ** 3
STRIDE = 1  # hardcoded per demo spec

# Model organisms for CoT generation (LoRA adapters loaded at startup).
MODEL_ORGANISMS: dict = {}  # disabled: only trained (Best v3) vs original_ao on public demo


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chat_compare_stripped")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def compute_layers(model_name, n_layers=None, layers=None):
    if layers:
        return [int(layer) for layer in layers]
    n = n_layers or 3
    percents = [int(100 * (i + 1) / (n + 1)) for i in range(n)]
    return [layer_percent_to_layer(model_name, p) for p in percents]


def load_dual_model(model_name, checkpoint_path, extra_checkpoints=None, organism_adapters=None, device="cuda"):
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {"device_map": "auto", "torch_dtype": dtype, "attn_implementation": choose_attn_implementation(model_name)}
    use_auto_8bit = device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < AUTO_8BIT_MEMORY_THRESHOLD
    if use_auto_8bit:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU has {total_gb:.1f} GiB total memory; enabling 8-bit base-model loading.")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        kwargs.pop("torch_dtype")
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"Loading trained CoT Oracle LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path, adapter_name="trained", is_trainable=False)
    ao_path = AO_CHECKPOINTS[model_name]
    print(f"Loading original AO from {ao_path}...")
    model.load_adapter(ao_path, adapter_name="original_ao", is_trainable=False)
    for name, path in (extra_checkpoints or {}).items():
        try:
            print(f"Loading extra checkpoint '{name}' from {path}...")
            model.load_adapter(path, adapter_name=name, is_trainable=False)
        except Exception as e:
            print(f"  WARNING: could not load extra checkpoint '{name}': {e}")
    loaded_organisms = []
    for adapter_name, adapter_info in (organism_adapters or {}).items():
        if adapter_info.get("type") != "lora":
            continue
        adapter_path = adapter_info["path"]
        print(f"Loading model organism '{adapter_name}' from {adapter_path}...")
        try:
            model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)
            loaded_organisms.append(adapter_name)
        except (ValueError, OSError) as e:
            print(f"  WARNING: Could not load organism '{adapter_name}': {e}")
    model.eval()
    print(f"  Adapters: {list(model.peft_config.keys())}")
    return model, tokenizer, loaded_organisms


def get_model_input_device(model):
    return model.get_input_embeddings().weight.device


def get_module_device(module):
    return next(module.parameters()).device


# ---------------------------------------------------------------------------
# Generation + oracle querying
# ---------------------------------------------------------------------------

def generate_cot_base(model, tokenizer, question, max_new_tokens=16384, cot_adapter=None, temperature=0.0):
    messages = [{"role": "user", "content": question}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(get_model_input_device(model))
    model.eval()
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        if cot_adapter and cot_adapter in model.peft_config:
            model.set_adapter(cot_adapter)
            output = model.generate(**inputs, **gen_kwargs)
        else:
            with model.disable_adapter():
                output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


def collect_multilayer_activations(model, tokenizer, text, layers, positions, cot_adapter=None):
    all_acts = []
    model.eval()
    input_device = str(get_model_input_device(model))
    for layer in layers:
        adapter_name = cot_adapter if cot_adapter and cot_adapter in model.peft_config else None
        if adapter_name:
            model.set_adapter(adapter_name)
        acts = collect_activations_at_positions(model, tokenizer, text, layer, positions, device=input_device, adapter_name=adapter_name)
        all_acts.append(acts.to("cpu"))
    return torch.cat(all_acts, dim=0)


def encode_prompt_with_positions(tokenizer, full_prompt, relative_spans):
    messages = [{"role": "user", "content": full_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    content_start = formatted.index(full_prompt)
    encoded = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    positions = []
    for rel_start, rel_end in relative_spans:
        abs_start = content_start + rel_start
        abs_end = content_start + rel_end
        token_positions = [i for i, (tok_start, tok_end) in enumerate(offsets) if tok_start < abs_end and tok_end > abs_start]
        if len(token_positions) != 1:
            raise ValueError(f"Expected exactly one token for span {(rel_start, rel_end)}, found {token_positions}")
        positions.append(token_positions[0])
    return input_ids, positions


def query_original_ao(model, tokenizer, acts_l50, prompt, model_name, max_new_tokens=150, temperature=0.0, adapter_name="original_ao", target_norm_scale=None):
    dtype = torch.bfloat16
    num_positions = acts_l50.shape[0]
    act_layer = layer_percent_to_layer(model_name, 50)
    prefix = f"L{act_layer}:" + SPECIAL_TOKEN * num_positions + ".\n"
    full_prompt = prefix + prompt
    label_len = len(f"L{act_layer}:")
    relative_spans = [(label_len + i * len(SPECIAL_TOKEN), label_len + (i + 1) * len(SPECIAL_TOKEN)) for i in range(num_positions)]
    input_ids, positions = encode_prompt_with_positions(tokenizer, full_prompt, relative_spans)
    input_tensor = torch.tensor([input_ids], device=get_model_input_device(model))
    attn_mask = torch.ones_like(input_tensor)
    model.set_adapter(adapter_name)
    injection_submodule = get_hf_submodule(model, 1, use_lora=True)
    hook_fn = get_steering_hook(vectors=acts_l50, positions=positions, device=get_module_device(injection_submodule), dtype=dtype, target_norm_scale=target_norm_scale)
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        output = model.generate(input_ids=input_tensor, attention_mask=attn_mask, **gen_kwargs)
    return tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)


def query_trained_oracle(model, tokenizer, selected_acts, prompt, selected_layers, layer_counts, max_new_tokens=150, temperature=0.0, adapter_name="trained", target_norm_scale=None):
    dtype = torch.bfloat16
    if len(selected_layers) != len(layer_counts):
        raise ValueError(f"selected_layers={selected_layers} and layer_counts={layer_counts} must align")
    total_count = sum(layer_counts)
    if selected_acts.shape[0] != total_count:
        raise ValueError(f"selected_acts rows {selected_acts.shape[0]} != expected {total_count}")
    # Match training-time prefix format from nl_probes/utils/dataset_utils.py
    # get_introspection_prefix(layers, num_positions):
    #   for layer in layers: prefix += f"Layer: {layer}\n" + SPECIAL_TOKEN*num_positions + "\n"
    prefix = ""
    relative_spans = []
    cursor = 0
    for layer, count in zip(selected_layers, layer_counts):
        label = f"Layer: {layer}\n"
        prefix += label
        cursor += len(label)
        for _ in range(count):
            start = cursor
            prefix += TRAINED_PLACEHOLDER
            cursor += len(TRAINED_PLACEHOLDER)
            relative_spans.append((start, cursor))
        # trailing newline ending the layer block (matches training prefix)
        prefix += "\n"
        cursor += 1
    full_prompt = prefix + prompt
    input_ids, positions = encode_prompt_with_positions(tokenizer, full_prompt, relative_spans)
    input_tensor = torch.tensor([input_ids], device=get_model_input_device(model))
    attn_mask = torch.ones_like(input_tensor)
    model.set_adapter(adapter_name)
    injection_submodule = get_hf_submodule(model, 1, use_lora=True)
    hook_fn = get_steering_hook(vectors=selected_acts, positions=positions, device=get_module_device(injection_submodule), dtype=dtype, target_norm_scale=target_norm_scale)
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad(), add_hook(injection_submodule, hook_fn):
        output = model.generate(input_ids=input_tensor, attention_mask=attn_mask, **gen_kwargs)
    return tokenizer.decode(output[0][len(input_ids):], skip_special_tokens=True)


def select_activation_cells(multilayer_acts, ao_acts, all_layers, n_positions_per_layer, selected_cells):
    n_layers = len(all_layers)
    d_model = multilayer_acts.shape[1]
    acts_by_layer = multilayer_acts.view(n_layers, n_positions_per_layer, d_model)
    if selected_cells is None:
        mask = torch.ones((n_layers, n_positions_per_layer), dtype=torch.bool)
    else:
        mask = torch.zeros((n_layers, n_positions_per_layer), dtype=torch.bool)
        layer_to_idx = {layer: idx for idx, layer in enumerate(all_layers)}
        skipped = 0
        for cell in selected_cells:
            layer = int(cell["layer"])
            pos = int(cell["position"])
            if layer not in layer_to_idx or not (0 <= pos < n_positions_per_layer):
                skipped += 1
                continue
            mask[layer_to_idx[layer]][pos] = True
        if skipped:
            log.warning(f"select_activation_cells: skipped {skipped}/{len(selected_cells)} out-of-range cells (likely stale browser session). n_layers={n_layers} n_positions={n_positions_per_layer}")
    if not torch.any(mask):
        raise ValueError("No activation cells selected")
    selected_chunks = []
    selected_layers = []
    layer_counts = []
    selected_positions = []
    for layer_idx, layer in enumerate(all_layers):
        pos_tensor = torch.nonzero(mask[layer_idx], as_tuple=False).flatten()
        if pos_tensor.numel() == 0:
            continue
        selected_layers.append(layer)
        layer_counts.append(int(pos_tensor.numel()))
        selected_chunks.append(acts_by_layer[layer_idx][pos_tensor])
        selected_positions.extend(int(pos.item()) for pos in pos_tensor)
    unique_positions = sorted(set(selected_positions))
    selected_multilayer = torch.cat(selected_chunks, dim=0)
    selected_ao = None if ao_acts is None else ao_acts[unique_positions]
    return selected_multilayer, selected_ao, selected_layers, layer_counts, unique_positions


def token_preview(tokenizer, token_id):
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n")
    if text.strip() == "":
        text = repr(text)[1:-1]
    if len(text) > 18:
        text = text[:15] + "..."
    return text


def decode_token_text(tokenizer, token_id):
    return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", " ")


def split_cot_answer(response_text):
    think_end = response_text.find("</think>")
    if think_end == -1:
        return response_text, ""
    cot_part = response_text[:think_end]
    if cot_part.startswith("<think>"):
        cot_part = cot_part[len("<think>"):]
    answer_part = response_text[think_end + len("</think>"):].strip()
    return cot_part.strip(), answer_part.strip()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    question: str = ""
    cot_response: str = ""
    cot_text: str = ""
    answer_text: str = ""
    full_text: str = ""
    stride_positions: list = field(default_factory=list)
    stride_token_ids: list = field(default_factory=list)
    token_labels: list = field(default_factory=list)
    cot_token_texts: list = field(default_factory=list)
    answer_token_texts: list = field(default_factory=list)
    sampled_token_to_stride_index: list = field(default_factory=list)
    multilayer_acts: torch.Tensor | None = None
    ao_acts: torch.Tensor | None = None
    prompt_len: int = 0
    cot_end: int = 0


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------

class ChatCompareWebApp:
    def __init__(self, args):
        self.args = args
        self.layers = compute_layers(args.model, n_layers=args.n_layers, layers=args.layers)
        self.layer_50 = layer_percent_to_layer(args.model, 50)
        extra = {}
        for item in (args.extra_checkpoints or []):
            if "=" not in item:
                continue
            name, path = item.split("=", 1)
            extra[name.strip()] = path.strip()
        self.model, self.tokenizer, self.organism_names = load_dual_model(
            args.model, args.checkpoint,
            extra_checkpoints=extra,
            organism_adapters=MODEL_ORGANISMS,
            device=args.device,
        )
        self._active_cot_adapter = args.cot_adapter or None
        self.state = SessionState()

        # --- Queue machinery ---
        # Global FIFO queue: any GPU-touching request acquires this lock.
        # `queue_waiting` tracks how many callers are waiting so we can
        # report a position in the queue to the frontend.
        self._queue_lock = asyncio.Lock()
        self._queue_waiting = 0
        self._queue_counter = 0  # total requests ever queued (for unique slot IDs)

        self.app = FastAPI(title="CoT Oracle Demo")
        self._register_routes()

    def _total_layers(self):
        from cot_utils import LAYER_COUNTS
        return LAYER_COUNTS.get(self.args.model)

    # --- Queueing helper ---

    async def _run_queued(self, fn, *fn_args, **fn_kwargs):
        """Run a (blocking) callable on a thread, gated by the global FIFO lock.

        Callers wait in FIFO order on self._queue_lock (asyncio.Lock grants in
        arrival order). While waiting, self._queue_waiting reflects how many
        requests are queued behind / alongside this one; /queue-status returns it.
        """
        self._queue_waiting += 1
        try:
            await self._queue_lock.acquire()
        except BaseException:
            self._queue_waiting -= 1
            raise
        self._queue_waiting -= 1
        try:
            return await asyncio.to_thread(fn, *fn_args, **fn_kwargs)
        finally:
            self._queue_lock.release()

    # --- GPU-touching operations ---

    def _compute_stride_info(self, full_text):
        all_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        messages = [{"role": "user", "content": self.state.question}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prompt_ids = self.tokenizer.encode(formatted, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        think_end_token = self.tokenizer.encode("</think>", add_special_tokens=False)
        cot_end = len(all_ids)
        for i in range(prompt_len, len(all_ids) - len(think_end_token) + 1):
            if all_ids[i:i + len(think_end_token)] == think_end_token:
                cot_end = i
                break
        # Sample stride positions over BOTH CoT and answer ranges
        cot_stride = get_cot_positions(prompt_len, cot_end, stride=STRIDE, tokenizer=self.tokenizer, input_ids=all_ids[:cot_end])
        answer_start_for_stride = cot_end + len(think_end_token) if cot_end < len(all_ids) else len(all_ids)
        if answer_start_for_stride < len(all_ids):
            answer_stride = get_cot_positions(answer_start_for_stride, len(all_ids), stride=STRIDE, tokenizer=self.tokenizer, input_ids=all_ids)
        else:
            answer_stride = []
        stride_positions = list(cot_stride) + list(answer_stride)
        if len(stride_positions) < 1:
            raise ValueError("Generation too short for any stride positions")
        stride_token_ids = [all_ids[pos] for pos in stride_positions]
        token_labels = [token_preview(self.tokenizer, tid) for tid in stride_token_ids]
        # Include both CoT and answer tokens in the rendered paragraph
        cot_token_ids = all_ids[prompt_len:]
        cot_token_texts = [decode_token_text(self.tokenizer, tid) for tid in cot_token_ids]
        answer_start = cot_end + len(think_end_token) if cot_end < len(all_ids) else len(all_ids)
        answer_token_ids = all_ids[answer_start:]
        answer_token_texts = [decode_token_text(self.tokenizer, tid) for tid in answer_token_ids]
        sampled_map = [None] * len(cot_token_ids)
        for stride_idx, full_pos in enumerate(stride_positions):
            rel = full_pos - prompt_len
            if 0 <= rel < len(sampled_map):
                sampled_map[rel] = stride_idx
        return {
            "prompt_len": prompt_len,
            "cot_end": cot_end,
            "stride_positions": stride_positions,
            "stride_token_ids": stride_token_ids,
            "token_labels": token_labels,
            "cot_token_texts": cot_token_texts,
            "answer_token_texts": answer_token_texts,
            "sampled_token_to_stride_index": sampled_map,
        }

    def _generate_and_extract(self, question, temperature):
        cot_adapter = self._active_cot_adapter
        cot_response = generate_cot_base(
            self.model, self.tokenizer, question,
            max_new_tokens=16384,
            cot_adapter=cot_adapter,
            temperature=temperature,
        )
        messages = [{"role": "user", "content": question}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        full_text = formatted + cot_response
        cot_text, answer_text = split_cot_answer(cot_response)
        self.state = SessionState(
            question=question,
            cot_response=cot_response,
            cot_text=cot_text,
            answer_text=answer_text,
            full_text=full_text,
        )
        info = self._compute_stride_info(full_text)
        stride_positions = info["stride_positions"]
        multilayer_acts = collect_multilayer_activations(
            self.model, self.tokenizer, full_text, self.layers, stride_positions,
            cot_adapter=cot_adapter,
        )
        input_device = str(get_model_input_device(self.model))
        ao_acts = collect_activations_at_positions(
            self.model, self.tokenizer, full_text, self.layer_50, stride_positions,
            device=input_device, adapter_name=None,
        )
        self.state.stride_positions = stride_positions
        self.state.stride_token_ids = info["stride_token_ids"]
        self.state.token_labels = info["token_labels"]
        self.state.cot_token_texts = info["cot_token_texts"]
        self.state.answer_token_texts = info["answer_token_texts"]
        self.state.sampled_token_to_stride_index = info["sampled_token_to_stride_index"]
        self.state.prompt_len = info["prompt_len"]
        self.state.cot_end = info["cot_end"]
        self.state.multilayer_acts = multilayer_acts
        self.state.ao_acts = ao_acts
        return {
            "question": question,
            "cot_response": cot_response,
            "cot_text": cot_text,
            "answer_text": answer_text,
            "stride_positions": stride_positions,
            "token_labels": info["token_labels"],
            "cot_token_texts": info["cot_token_texts"],
            "answer_token_texts": info["answer_token_texts"],
            "sampled_token_to_stride_index": info["sampled_token_to_stride_index"],
            "layers": self.layers,
            "layer_50": self.layer_50,
            "total_layers": self._total_layers(),
            "n_positions": len(stride_positions),
            "n_vectors": int(multilayer_acts.shape[0]),
        }

    def _resolve_context(self, prompt, selected_cells):
        if self.state.multilayer_acts is None:
            raise HTTPException(status_code=400, detail="Generate a CoT first")
        prompt = (prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is empty")
        selected_ml, selected_ao, selected_layers, layer_counts, selected_positions = select_activation_cells(
            self.state.multilayer_acts,
            self.state.ao_acts,
            self.layers,
            len(self.state.stride_positions),
            selected_cells if selected_cells else None,
        )
        return {
            "prompt": prompt,
            "selected_ml": selected_ml,
            "selected_ao": selected_ao,
            "selected_layers": selected_layers,
            "layer_counts": layer_counts,
            "selected_positions": selected_positions,
        }

    def _run_both(self, prompt, selected_cells, max_tokens, temperature):
        ctx = self._resolve_context(prompt, selected_cells)
        # Per-adapter post-injection norm rescale (matches training-time
        # AO_FINAL_NORM_SCALE env var). None = natural ~√2× additive.
        scales = getattr(self.args, "_target_norm_scales", {}) or {}
        ao_response = query_original_ao(
            self.model, self.tokenizer, ctx["selected_ao"], ctx["prompt"],
            model_name=self.args.model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            target_norm_scale=scales.get("original_ao"),
        )
        trained_response = query_trained_oracle(
            self.model, self.tokenizer, ctx["selected_ml"], ctx["prompt"],
            ctx["selected_layers"], ctx["layer_counts"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            target_norm_scale=scales.get("trained"),
        )
        return {
            "prompt": ctx["prompt"],
            "ao_response": ao_response,
            "trained_response": trained_response,
            "selected_layers": ctx["selected_layers"],
            "selected_positions": ctx["selected_positions"],
        }

    # --- Routes ---

    def _register_routes(self):
        @self.app.middleware("http")
        async def _error_logging_middleware(request: Request, call_next):
            try:
                return await call_next(request)
            except Exception as exc:
                tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                log.error("Unhandled exception on %s:\n%s", request.url.path, tb_str)
                return JSONResponse(status_code=500, content={"detail": tb_str})

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse(self._render_html())

        @self.app.get("/api/config")
        async def config():
            from cot_utils import LAYER_COUNTS
            total_layers = LAYER_COUNTS.get(self.args.model)
            return {
                "layers": self.layers,
                "layer_50": self.layer_50,
                "total_layers": total_layers,
                "stride": STRIDE,
                "organisms": [{"key": n, "label": MODEL_ORGANISMS[n]["label"]} for n in self.organism_names],
            }

        @self.app.get("/queue-status")
        async def queue_status():
            return {"waiting": self._queue_waiting, "locked": self._queue_lock.locked()}

        @self.app.post("/api/generate")
        async def generate(payload: dict):
            question = (payload.get("question") or "").strip()
            if not question:
                raise HTTPException(status_code=400, detail="Question is empty")
            temperature = float(payload.get("temperature", 0))
            return await self._run_queued(self._generate_and_extract, question, temperature)

        @self.app.post("/api/run")
        async def run(payload: dict):
            prompt = payload.get("prompt", "")
            selected_cells = payload.get("selected_cells", [])
            max_tokens = int(payload.get("max_tokens", self.args.max_tokens))
            temperature = float(payload.get("oracle_temperature", 0))
            return await self._run_queued(self._run_both, prompt, selected_cells, max_tokens, temperature)

    # --- HTML template ---

    def _render_html(self):
        return HTML_TEMPLATE


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>AO demo</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding-bottom: 48px; background: #0f172a; color: #e2e8f0; }
    header { padding: 18px 22px 8px 22px; border-bottom: 1px solid #1e293b; background: #0b1220; }
    header h1 { margin: 0 0 4px 0; font-size: 22px; color: #e2e8f0; }
    header .sub { font-size: 13px; color: #94a3b8; }
    header .how-to { margin: 10px 0 0 0; padding-left: 22px; font-size: 13px; color: #cbd5e1; }
    header .how-to li { margin-bottom: 3px; line-height: 1.45; }
    header .how-to b { color: #e2e8f0; }
    header a { color: #60a5fa; text-decoration: none; }
    header a:hover { text-decoration: underline; }
    .page { display: grid; grid-template-columns: 340px 1fr; min-height: calc(100vh - 80px); }
    .sidebar, .main { padding: 16px; }
    .sidebar { border-right: 1px solid #334155; background: #111827; }
    textarea, select, input { width: 100%; box-sizing: border-box; background: #0f172a; color: #e2e8f0; border: 1px solid #475569; border-radius: 8px; padding: 8px; font-family: inherit; }
    textarea { min-height: 100px; resize: vertical; font-size: 13px; }
    button { background: #2563eb; color: white; border: 0; border-radius: 8px; padding: 10px 12px; cursor: pointer; font-family: inherit; }
    button.secondary { background: #334155; }
    button:disabled { opacity: 0.5; cursor: default; }
    .row { display: flex; gap: 8px; margin-top: 8px; }
    .row > * { flex: 1; }
    .status { margin-top: 12px; font-size: 13px; color: #93c5fd; min-height: 20px; }
    .muted { color: #94a3b8; font-size: 12px; }
    .panel { background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 12px; margin-bottom: 12px; }
    .token-wrap { border: 1px solid #334155; border-radius: 12px; background: #020617; padding: 12px; }
    .layer-block { border: 1px solid #1e293b; border-radius: 12px; padding: 10px; background: #0b1220; margin-bottom: 10px; }
    .layer-label { display: inline-block; margin-bottom: 8px; padding: 4px 10px; border-radius: 999px; background: #1e293b; font-weight: 700; font-size: 12px; }
    .token-paragraph { white-space: pre-wrap; line-height: 1.9; user-select: none; font-size: 13px; }
    .tok { border-radius: 6px; padding: 1px 2px; position: relative; }
    .tok.sampled { cursor: pointer; background: rgba(51, 65, 85, 0.35); }
    .tok.sampled:hover { background: rgba(96, 165, 250, 0.22); }
    .tok.sampled.selected { background: #1d4ed8; color: #eff6ff; }
    .tok.unsampled { color: #64748b; }
    .selection-box { position: fixed; border: 1px solid #60a5fa; background: rgba(96,165,250,0.14); pointer-events: none; display: none; z-index: 50; }
    .outputs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .text-block { white-space: pre-wrap; word-break: break-word; line-height: 1.5; font-size: 13px; }
    .small { font-size: 12px; }
    footer { position: fixed; bottom: 0; left: 0; right: 0; padding: 8px 16px; font-size: 11px; color: #64748b; background: #0b1220; border-top: 1px solid #1e293b; text-align: center; }
    label { font-size: 12px; color: #cbd5e1; }
    .slider-row { display: flex; align-items: center; gap: 8px; margin-top: 6px; }
    .slider-row input[type=range] { flex: 1; }
    .slider-row span { min-width: 36px; font-variant-numeric: tabular-nums; color: #94a3b8; font-size: 12px; }
    h3 { margin: 0 0 8px 0; font-size: 14px; color: #e2e8f0; }
    .queue-banner { display: none; background: #78350f; color: #fef3c7; padding: 8px 12px; border-radius: 8px; margin-bottom: 8px; font-size: 13px; }
    .queue-banner.show { display: block; }
  </style>
</head>
<body>
  <header>
    <h1>Play with the models from "Building Better Activation Oracles"</h1>
    <ol class="how-to">
      <li><b>Ask a question.</b> The model generates a chain of thought; we record its internal activations while it thinks.</li>
      <li><b>Select tokens.</b> Click or drag across the chain of thought to pick which positions the oracle gets to peek at.</li>
      <li><b>Ask the oracle.</b> Type a question about the model's reasoning (e.g. "is it confident?", "what answer is it heading toward?"). Both Adam Original and Ours answer using only the selected activations &mdash; no access to the CoT text.</li>
    </ol>
  </header>
  <div class="page">
    <div class="sidebar">
      <div id="queueBanner" class="queue-banner"></div>
      <label>1. Question</label>
      <textarea id="question" placeholder="Ask the model to think..."></textarea>
      <label style="margin-top:10px;display:block" class="small">Generator temperature <span class="muted" id="genTempVal">0.0</span></label>
      <div class="slider-row"><input id="genTemp" type="range" min="0" max="1.5" step="0.05" value="0"><span id="genTempReadout">0.00</span></div>
      <div class="row"><button id="generateBtn">Generate CoT + Activations</button></div>
      <div class="status" id="status"></div>

      <div class="panel" style="margin-top:16px">
        <label>3. Oracle prompt</label>
        <textarea id="oraclePrompt" placeholder="enter your prompt here"></textarea>
        <label style="margin-top:10px;display:block" class="small">Oracle temperature <span class="muted" id="oracleTempVal">0.0</span></label>
        <div class="slider-row"><input id="oracleTemp" type="range" min="0" max="1.5" step="0.05" value="0"><span id="oracleTempReadout">0.00</span></div>
        <div class="row">
          <button id="runBtn">Run both oracles</button>
        </div>
        <div class="row">
          <button id="selectAllBtn" class="secondary">Select all</button>
          <button id="clearBtn" class="secondary">Clear selection</button>
        </div>
        <div class="small muted" id="selectionInfo" style="margin-top:8px">No session yet.</div>
      </div>
    </div>

    <div class="main">
      <div class="panel">
        <div id="meta" class="muted">Generate a CoT to populate the selectable activation text.</div>
      </div>
      <div class="panel">
        <h3>2. Chain of Thought &mdash; select tokens</h3>
        <div class="muted">Click or drag across highlighted tokens.</div>
        <div id="tokenRowsWrap" class="token-wrap" style="margin-top:10px"></div>
      </div>

      <details class="panel" style="padding:0">
        <summary style="cursor:pointer;padding:14px 16px;font-weight:600;color:#e2e8f0">Answer (click to expand)</summary>
        <div id="answerPreview" class="text-block" style="max-height:240px;overflow-y:auto;padding:0 16px 14px 16px"></div>
      </details>

      <div class="outputs">
        <div class="panel">
          <h3>Adam Original</h3>
          <div id="aoOutput" class="text-block muted">Run the oracle to see output.</div>
        </div>
        <div class="panel">
          <h3>Ours</h3>
          <div id="trainedOutput" class="text-block muted">Run the oracle to see output.</div>
        </div>
      </div>
    </div>
  </div>

  <div class="selection-box" id="selBox"></div>
  <footer>Demo running on a single RTX 6000 Ada \u2014 please be patient if queued.</footer>

  <script>
    const genTemp = document.getElementById('genTemp');
    const genTempReadout = document.getElementById('genTempReadout');
    genTemp.addEventListener('input', () => { genTempReadout.textContent = Number(genTemp.value).toFixed(2); });
    const oracleTemp = document.getElementById('oracleTemp');
    const oracleTempReadout = document.getElementById('oracleTempReadout');
    oracleTemp.addEventListener('input', () => { oracleTempReadout.textContent = Number(oracleTemp.value).toFixed(2); });

    const statusEl = document.getElementById('status');
    const queueBanner = document.getElementById('queueBanner');

    let sessionData = null;    // response from /api/generate
    let selectedPositions = new Set();  // stride positions

    function setStatus(msg) { statusEl.textContent = msg; }

    async function pollQueueWhileBusy(cancelSignal) {
      while (!cancelSignal.done) {
        try {
          const r = await fetch('/queue-status');
          const j = await r.json();
          if (j.waiting > 0 || j.locked) {
            queueBanner.classList.add('show');
            queueBanner.textContent = 'Your request is queued. ' + j.waiting + ' ahead of you; GPU busy.';
          } else {
            queueBanner.classList.remove('show');
          }
        } catch(e) {}
        await new Promise(r => setTimeout(r, 1000));
      }
      queueBanner.classList.remove('show');
    }

    async function withQueueUi(fn) {
      const cancel = { done: false };
      pollQueueWhileBusy(cancel);
      try {
        return await fn();
      } finally {
        cancel.done = true;
        queueBanner.classList.remove('show');
      }
    }

    function renderTokens() {
      const wrap = document.getElementById('tokenRowsWrap');
      wrap.innerHTML = '';
      if (!sessionData) { wrap.innerHTML = '<div class="muted">No session yet.</div>'; return; }
      const sampledMap = sessionData.sampled_token_to_stride_index || [];
      const cotTokenTexts = sessionData.cot_token_texts || [];
      const para = document.createElement('div');
      para.className = 'token-paragraph';
      for (let i = 0; i < cotTokenTexts.length; i++) {
        const t = cotTokenTexts[i];
        const strideIdx = sampledMap[i];
        const span = document.createElement('span');
        if (strideIdx === null || strideIdx === undefined) {
          span.className = 'tok unsampled';
        } else {
          span.className = 'tok sampled';
          span.dataset.position = strideIdx;
          if (selectedPositions.has(strideIdx)) span.classList.add('selected');
        }
        span.textContent = t;
        para.appendChild(span);
      }
      wrap.appendChild(para);
      updateSelectionInfo();
    }

    function updateSelectionInfo() {
      const info = document.getElementById('selectionInfo');
      if (!sessionData) { info.textContent = 'No session yet.'; return; }
      info.textContent = 'Selected ' + selectedPositions.size + ' / ' + (sessionData.n_positions || 0) + ' tokens.';
    }

    // --- Drag selection ---
    const selBox = document.getElementById('selBox');
    let dragging = false, dragStart = null, dragAdditive = true, dragInitialSet = null;

    document.addEventListener('mousedown', (ev) => {
      const wrap = document.getElementById('tokenRowsWrap');
      if (!wrap.contains(ev.target)) return;
      // allow normal clicks on non-sampled tokens; start drag on any sampled area
      dragging = true;
      dragStart = { x: ev.clientX, y: ev.clientY };
      dragAdditive = !ev.shiftKey; // shift removes, default adds
      dragInitialSet = new Set(selectedPositions);
      selBox.style.left = ev.clientX + 'px'; selBox.style.top = ev.clientY + 'px';
      selBox.style.width = '0px'; selBox.style.height = '0px';
      selBox.style.display = 'block';
      ev.preventDefault();
    });

    document.addEventListener('mousemove', (ev) => {
      if (!dragging) return;
      const x1 = Math.min(dragStart.x, ev.clientX);
      const y1 = Math.min(dragStart.y, ev.clientY);
      const x2 = Math.max(dragStart.x, ev.clientX);
      const y2 = Math.max(dragStart.y, ev.clientY);
      selBox.style.left = x1 + 'px';
      selBox.style.top = y1 + 'px';
      selBox.style.width = (x2 - x1) + 'px';
      selBox.style.height = (y2 - y1) + 'px';

      const newSel = new Set(dragInitialSet);
      document.querySelectorAll('.tok.sampled').forEach((el) => {
        const r = el.getBoundingClientRect();
        const hit = r.left < x2 && r.right > x1 && r.top < y2 && r.bottom > y1;
        if (hit) {
          const k = Number(el.dataset.position);
          if (dragAdditive) newSel.add(k); else newSel.delete(k);
        }
      });
      selectedPositions = newSel;
      document.querySelectorAll('.tok.sampled').forEach((el) => {
        const k = Number(el.dataset.position);
        el.classList.toggle('selected', selectedPositions.has(k));
      });
      updateSelectionInfo();
    });

    document.addEventListener('mouseup', (ev) => {
      if (!dragging) return;
      dragging = false;
      selBox.style.display = 'none';
      const dx = Math.abs(ev.clientX - dragStart.x);
      const dy = Math.abs(ev.clientY - dragStart.y);
      if (dx < 4 && dy < 4) {
        // Treat as a click: toggle the token under the pointer
        const el = ev.target.closest('.tok.sampled');
        if (el) {
          const k = Number(el.dataset.position);
          if (selectedPositions.has(k)) selectedPositions.delete(k); else selectedPositions.add(k);
          el.classList.toggle('selected', selectedPositions.has(k));
          updateSelectionInfo();
        }
      }
    });

    document.getElementById('selectAllBtn').addEventListener('click', () => {
      if (!sessionData) return;
      selectedPositions = new Set();
      for (let p = 0; p < (sessionData.n_positions || 0); p++) selectedPositions.add(p);
      renderTokens();
    });

    document.getElementById('clearBtn').addEventListener('click', () => {
      selectedPositions = new Set();
      renderTokens();
    });

    document.getElementById('generateBtn').addEventListener('click', async () => {
      const question = document.getElementById('question').value.trim();
      if (!question) { setStatus('Enter a question first.'); return; }
      setStatus('Generating CoT + activations...');
      document.getElementById('generateBtn').disabled = true;
      try {
        const result = await withQueueUi(async () => {
          const resp = await fetch('/api/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ question, temperature: Number(genTemp.value) }),
          });
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
          }
          return await resp.json();
        });
        sessionData = result;
        selectedPositions = new Set();
        for (let p = 0; p < (result.n_positions || 0); p++) selectedPositions.add(p);
        document.getElementById('answerPreview').textContent = result.answer_text || '(no answer)';
        document.getElementById('meta').textContent =
          'Question: ' + result.question + '  \u00b7  ' + result.n_positions + ' selectable tokens';
        renderTokens();
        setStatus('Ready. Drag to (de)select activation cells, then click Run both oracles.');
      } catch(e) {
        setStatus('Error: ' + e.message);
      } finally {
        document.getElementById('generateBtn').disabled = false;
      }
    });

    document.getElementById('runBtn').addEventListener('click', async () => {
      if (!sessionData) { setStatus('Generate a CoT first.'); return; }
      const prompt = document.getElementById('oraclePrompt').value.trim();
      if (!prompt) { setStatus('Enter an oracle prompt first.'); return; }
      const layers = (sessionData.layers || []);
      const cells = [];
      Array.from(selectedPositions).forEach(p => {
        layers.forEach(L => cells.push({ layer: Number(L), position: Number(p) }));
      });
      if (cells.length === 0) { setStatus('Select at least one token.'); return; }
      setStatus('Querying oracles...');
      document.getElementById('runBtn').disabled = true;
      document.getElementById('aoOutput').textContent = '...';
      document.getElementById('trainedOutput').textContent = '...';
      try {
        const result = await withQueueUi(async () => {
          const resp = await fetch('/api/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              prompt,
              selected_cells: cells,
              oracle_temperature: Number(oracleTemp.value),
            }),
          });
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || resp.statusText);
          }
          return await resp.json();
        });
        document.getElementById('aoOutput').textContent = result.ao_response || '(empty)';
        document.getElementById('trainedOutput').textContent = result.trained_response || '(empty)';
        document.getElementById('aoOutput').classList.remove('muted');
        document.getElementById('trainedOutput').classList.remove('muted');
        setStatus('Done.');
      } catch(e) {
        setStatus('Error: ' + e.message);
      } finally {
        document.getElementById('runBtn').disabled = false;
      }
    });

  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="CoT Oracle public demo")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", required=True, help="Trained LoRA checkpoint path or HF repo")
    parser.add_argument("--cot-adapter", default=None, help="Model organism LoRA for CoT generation")
    parser.add_argument("--n-layers", type=int, default=None, help="Number of evenly-spaced layers")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Explicit layer indices (overrides --n-layers)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--host", default="127.0.0.1", help="Web host")
    parser.add_argument("--port", type=int, default=8000, help="Web port")
    parser.add_argument("--extra-checkpoints", nargs="+", default=[], metavar="NAME=PATH",
                        help="Extra oracle LoRA checkpoints as name=path pairs")
    parser.add_argument("--original-ao-override", default=None,
                        help="Override the hardcoded original_ao LoRA (AO_CHECKPOINTS[model]) "
                             "with a different HF repo or local path. Use to put e.g. Best v3 "
                             "in the 'left' side of the side-by-side instead of Adam's reference.")
    parser.add_argument("--target-norm-scales", nargs="+", default=[], metavar="NAME=SCALE",
                        help="Per-adapter post-injection norm rescale, mirroring training-time "
                             "AO_FINAL_NORM_SCALE. e.g. --target-norm-scales trained=2.0 keeps "
                             "the left/original side at natural ~√2× and rescales the right "
                             "(trained) side's injection to 2.0×‖orig‖.")
    return parser


def main():
    args = build_parser().parse_args()
    # Override AO_CHECKPOINTS so 'original_ao' can be a different LoRA than the
    # default Adam reference. Done before ChatCompareWebApp instantiation so
    # load_dual_model picks it up.
    if args.original_ao_override:
        from core import ao as _ao_mod
        _ao_mod.AO_CHECKPOINTS[args.model] = args.original_ao_override
        print(f"[override] AO_CHECKPOINTS[{args.model}] = {args.original_ao_override}")
    # Parse per-adapter target norm scales.
    scales: dict[str, float] = {}
    for item in (args.target_norm_scales or []):
        if "=" not in item:
            continue
        name, scale = item.split("=", 1)
        scales[name.strip()] = float(scale)
    args._target_norm_scales = scales
    if scales:
        print(f"[target-norm-scales] {scales}")
    web_app = ChatCompareWebApp(args)
    print(f"Serving CoT Oracle demo on http://{args.host}:{args.port}")
    uvicorn.run(web_app.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
