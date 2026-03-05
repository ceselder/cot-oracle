"""
Add source column from cot-oracle-corpus-v5 to cleaned datasets and create train/test splits.
Splits: 30% of source categories held out in test, ~70/30 example ratio.
Publishes to japhba HuggingFace org.
"""
import itertools
import collections
import hashlib
from datasets import load_dataset, Dataset, DatasetDict

HF_SRC_ORG = "mats-10-sprint-cs-jb"
HF_DST_ORG = "japhba"

# Load corpus for question→source mapping
print("Loading corpus-v5...")
corpus = load_dataset(f"{HF_SRC_ORG}/cot-oracle-corpus-v5", split="train")
question_to_source = {row["question"]: row["source"] for row in corpus}
print(f"  Corpus size: {len(corpus)}, unique sources: {len(set(corpus['source']))}")


def infer_source_heuristic(question: str) -> str:
    """Heuristic for RT questions not in corpus-v5 — all treated as lmsys_neutral
    since that source contains the same diverse chat/safety question patterns."""
    return "lmsys_neutral"


def add_source_column(rows, task_name):
    """Add 'corpus_source' field to each row by joining with corpus-v5."""
    result = []
    n_matched = 0
    for row in rows:
        row = dict(row)
        q = row.get("question", "")
        if q in question_to_source:
            row["source"] = question_to_source[q]
            n_matched += 1
        else:
            row["source"] = infer_source_heuristic(q)
        result.append(row)
    print(f"  {task_name}: {n_matched}/{len(rows)} matched to corpus-v5 ({n_matched/len(rows)*100:.1f}%)")
    return result


def best_holdout_sources(source_counts: dict, target_frac: float = 0.30) -> list[str]:
    """Select sources to hold out such that:
    - We hold out floor(0.3 * n_sources) categories (at least 1)
    - The held-out example count is as close to target_frac of total as possible
    """
    total = sum(source_counts.values())
    target_n = total * target_frac
    n_sources = len(source_counts)
    n_holdout = max(1, round(n_sources * target_frac))
    print(f"  {n_sources} sources → holding out {n_holdout} (={n_holdout/n_sources*100:.0f}% of categories)")

    sources = list(source_counts.keys())
    best_combo = None
    best_diff = float("inf")
    for combo in itertools.combinations(sources, n_holdout):
        held_n = sum(source_counts[s] for s in combo)
        diff = abs(held_n - target_n)
        if diff < best_diff:
            best_diff = diff
            best_combo = combo

    held_n = sum(source_counts[s] for s in best_combo)
    print(f"  Held-out sources: {sorted(best_combo)}")
    print(f"  Held-out examples: {held_n}/{total} ({held_n/total*100:.1f}%)")
    return list(best_combo)


def make_split(rows: list[dict], held_out_sources: list[str]) -> DatasetDict:
    train = [r for r in rows if r["source"] not in held_out_sources]
    test = [r for r in rows if r["source"] in held_out_sources]
    return DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)})


# ── answer_trajectory ─────────────────────────────────────────────────────────
print("\n=== answer_trajectory ===")
at = load_dataset(f"{HF_SRC_ORG}/cot-oracle-answer-trajectory-cleaned", split="train")
print(f"  Loaded {len(at)} rows")
at_rows = add_source_column(list(at), "answer_trajectory")
at_src_counts = collections.Counter(r["source"] for r in at_rows)
print(f"  Source distribution: {dict(at_src_counts.most_common())}")
at_held = best_holdout_sources(at_src_counts)
at_ds = make_split(at_rows, at_held)
print(f"  Split: {len(at_ds['train'])} train / {len(at_ds['test'])} test")

# ── atypical_answer ───────────────────────────────────────────────────────────
print("\n=== atypical_answer ===")
aa = load_dataset(f"{HF_SRC_ORG}/cot-oracle-atypical-answer-cleaned", split="train")
print(f"  Loaded {len(aa)} rows")
# Already has fine-grained source column (mmlu_pro_* subcategories)
aa_rows = [dict(r) for r in aa]
# Verify source column exists
assert all("source" in r for r in aa_rows), "atypical_answer missing source column"
aa_src_counts = collections.Counter(r["source"] for r in aa_rows)
print(f"  Source distribution: {dict(aa_src_counts.most_common())}")
aa_held = best_holdout_sources(aa_src_counts)
aa_ds = make_split(aa_rows, aa_held)
print(f"  Split: {len(aa_ds['train'])} train / {len(aa_ds['test'])} test")

# ── reasoning_termination ─────────────────────────────────────────────────────
print("\n=== reasoning_termination ===")
rt = load_dataset(f"{HF_SRC_ORG}/cot-oracle-reasoning-termination-cleaned", split="train")
print(f"  Loaded {len(rt)} rows")
rt_rows = add_source_column(list(rt), "reasoning_termination")
rt_src_counts = collections.Counter(r["source"] for r in rt_rows)
print(f"  Source distribution: {dict(rt_src_counts.most_common())}")
rt_held = best_holdout_sources(rt_src_counts)
rt_ds = make_split(rt_rows, rt_held)
print(f"  Split: {len(rt_ds['train'])} train / {len(rt_ds['test'])} test")

# ── Push to HuggingFace ───────────────────────────────────────────────────────
DATASETS = [
    ("cot-oracle-answer-trajectory", at_ds, at_held),
    ("cot-oracle-atypical-answer", aa_ds, aa_held),
    ("cot-oracle-reasoning-termination", rt_ds, rt_held),
]

print("\n=== Pushing to HuggingFace ===")
for name, ds_dict, held in DATASETS:
    repo = f"{HF_DST_ORG}/{name}"
    print(f"  Pushing {repo} ...")
    ds_dict.push_to_hub(repo, private=False)
    print(f"    Done. train={len(ds_dict['train'])}, test={len(ds_dict['test'])}")
    print(f"    Held-out sources: {held}")

print("\nAll done!")
