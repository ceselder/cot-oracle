#!/usr/bin/env python3
"""Quick test of all Stage 1+2 data loaders."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.ao_repo import ensure_ao_repo_on_path
ensure_ao_repo_on_path()
from nl_probes.utils.common import load_tokenizer

tok = load_tokenizer("Qwen/Qwen3-8B")
CORPUS = "data/cot_corpus_v5/corpus_medium.jsonl"
MODEL = "Qwen/Qwen3-8B"
N = 10

# Stage 1
from dataset_classes.cot_rollout_multilayer import load_cot_rollout_multilayer
print("--- Full Reconstruction ---")
d = load_cot_rollout_multilayer(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, num_positions={d[0]['num_positions']}, layers={d[0].get('layers')}")

from dataset_classes.cot_next_step import load_cot_next_step_data
print("--- Next Step ---")
d = load_cot_next_step_data(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, target='{d[0]['target_response'][:50]}'")

from dataset_classes.cot_answer_prediction import load_cot_answer_prediction_data
print("--- Answer Prediction ---")
d = load_cot_answer_prediction_data(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, target='{d[0]['target_response'][:50]}'")

from dataset_classes.cot_load_bearing import load_cot_load_bearing_data
print("--- Load Bearing ---")
d = load_cot_load_bearing_data(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, target='{d[0]['target_response']}'")

# Stage 2
from dataset_classes.cot_correctness import load_cot_correctness_data
print("--- Correctness ---")
d = load_cot_correctness_data(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, target='{d[0]['target_response']}'")

from dataset_classes.cot_decorative import load_cot_decorative_data
print("--- Decorative ---")
d = load_cot_decorative_data(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, target='{d[0]['target_response']}'")

from dataset_classes.cot_domain import load_cot_domain_data
print("--- Domain ---")
d = load_cot_domain_data(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, target='{d[0]['target_response']}'")

from dataset_classes.cot_importance import load_cot_importance_data
print("--- Importance ---")
d = load_cot_importance_data(CORPUS, tok, MODEL, num_examples=N, stride=5)
print(f"  Got {len(d)}, target='{d[0]['target_response']}'")

print("\nAll loaders passed!")
