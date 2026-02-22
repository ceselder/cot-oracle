"""Precompute boundary_positions for a corpus JSONL file. Rewrites in-place."""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from transformers import AutoTokenizer
from cot_utils import split_cot_into_sentences, find_sentence_boundary_positions
from tqdm.auto import tqdm

corpus_path = sys.argv[1]
model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

entries = []
with open(corpus_path) as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

needs = sum(1 for e in entries if not e.get("boundary_positions"))
print(f"{len(entries)} entries, {needs} need boundary_positions", flush=True)
if needs == 0:
    print("All entries already have boundary_positions, skipping.")
    sys.exit(0)

for entry in tqdm(entries, desc="Computing boundaries"):
    if entry.get("boundary_positions"):
        continue
    messages = [{"role": "user", "content": entry["question"]}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    cot_text = entry["cot_response"]
    think_end = cot_text.find("</think>")
    if think_end != -1:
        cot_text = cot_text[:think_end]
    sentences = split_cot_into_sentences(cot_text)
    positions = find_sentence_boundary_positions(tokenizer, formatted + cot_text, sentences)
    entry["boundary_positions"] = positions

print("Writing...", flush=True)
with open(corpus_path, "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")
print("Done!", flush=True)
