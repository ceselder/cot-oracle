"""Pre-build and cache training data for train_random_layers.py."""
import sys, os, time, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ao_reference"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from collections import defaultdict
from nl_probes.utils.common import load_tokenizer
from train_random_layers import build_training_mixture, build_eval_datasets
from corpus_tokenize import load_corpus, pretokenize_corpus, ensure_boundary_positions
from data_cache import load_cached_data, save_cached_data

model_name = "Qwen/Qwen3-8B"
corpus = "data/cot_corpus_v5/corpus.jsonl"
layer_percents = [25, 50, 75]
max_layers = 36
layer_mean = 5
position_strides = [4, 16, 64]
max_positions = 50
num_workers = 8

task_sizes = {
    "cot_context_prediction": 100000,
    "cot_sentence_prediction": 30000,
    "cot_decorative": 10000,
    "cot_domain": 15000,
    "cot_correctness": 15000,
    "cot_persona": 0,
}

cache_extra = dict(
    layer_mean=layer_mean, max_layers=max_layers, layer_percents=layer_percents,
    position_strides=position_strides, max_positions=max_positions, labels_dir=None,
)

cached = load_cached_data(corpus, None, task_sizes, model_name, **cache_extra)
if cached is not None:
    print("Already cached!")
    sys.exit(0)

# Ensure boundary positions + pre-tokenize once (parallel)
print("Ensuring boundary_positions...")
corpus_entries = ensure_boundary_positions(corpus, model_name, num_workers=num_workers)
print("Pre-tokenizing corpus...")
pretokenize_corpus(corpus_entries, model_name, num_workers=num_workers)

tokenizer = load_tokenizer(model_name)

t0 = time.time()
random.seed(42)
training_data = build_training_mixture(
    corpus, None, None, tokenizer, model_name, layer_percents,
    max_layers, layer_mean, task_sizes, position_strides, max_positions,
    num_workers=num_workers, corpus_entries=corpus_entries,
)
assert training_data, "No training data!"

eval_datasets = build_eval_datasets(
    corpus, None, tokenizer, model_name, layer_percents,
    max_layers, layer_mean, position_strides, max_positions,
    corpus_entries=corpus_entries,
)

by_type = defaultdict(list)
for dp in training_data:
    by_type[dp.datapoint_type].append(dp)

final_training = []
for dtype, dps in by_type.items():
    if len(dps) > 100:
        eval_datasets[dtype] = dps[-100:]
        final_training.extend(dps[:-100])
    else:
        final_training.extend(dps)

elapsed = time.time() - t0
print(f"\nTotal: {len(final_training)} training, {sum(len(v) for v in eval_datasets.values())} eval in {elapsed:.0f}s")

save_cached_data(final_training, eval_datasets, corpus, None, task_sizes, model_name, **cache_extra)
print("Done!")
