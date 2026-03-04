#!/usr/bin/env python3
"""
Generate unified CoT description eval dataset with calibration.

4-tier dataset: entities, quantities, relationships, multi_detail.
Each text produces an answerable + unanswerable query pair.

3-round pipeline using Gemini 3.1 Flash Lite via OpenRouter:
  Round 1: Generate texts + query pairs (answerable + unanswerable) per tier
  Round 2: Generate reference answers for both answerable and unanswerable items
  Round 3: (at eval time — handled by eval_loop.py, not this script)

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_hallucination_vagueness_eval.py
    python scripts/generate_hallucination_vagueness_eval.py --n 1000 --max-budget 3.0 --dry-run
    python scripts/generate_hallucination_vagueness_eval.py --n 1000 --max-budget 3.0 --upload
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import httpx

# ── Model config ──

MODEL_ID = "google/gemini-3.1-flash-lite-preview"
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
CONCURRENCY = 30

# ── Stats ──

completed = 0
failed = 0
total_input_tokens = 0
total_output_tokens = 0
budget_exceeded = False


def estimate_cost():
    return total_input_tokens * INPUT_COST_PER_M / 1e6 + total_output_tokens * OUTPUT_COST_PER_M / 1e6


# ── API call ──

async def call_openrouter(
    client: httpx.AsyncClient,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_budget: float,
    max_tokens: int = 800,
) -> tuple[str, int, int]:
    global completed, failed, budget_exceeded, total_input_tokens, total_output_tokens

    if budget_exceeded:
        return "", 0, 0

    async with semaphore:
        body = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for attempt in range(4):
            try:
                resp = await client.post(ENDPOINT, json=body, headers=headers, timeout=90)

                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt + random.random())
                    continue
                if resp.status_code == 402:
                    budget_exceeded = True
                    print("\n*** Budget exceeded (402). Stopping. ***")
                    return "", 0, 0
                if resp.status_code != 200:
                    if attempt == 3:
                        failed += 1
                        if failed <= 10:
                            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                        return "", 0, 0
                    await asyncio.sleep(2 ** attempt)
                    continue

                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                usage = data.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                total_input_tokens += in_tok
                total_output_tokens += out_tok

                completed += 1
                if completed % 50 == 0:
                    print(f"  {completed} calls done, {failed} failed, ~${estimate_cost():.3f} spent")

                current_cost = estimate_cost()
                if current_cost > max_budget * 0.95:
                    budget_exceeded = True
                    print(f"\n*** Approaching budget limit (${current_cost:.2f}/${max_budget:.2f}). Stopping. ***")

                return content, in_tok, out_tok

            except (httpx.TimeoutException, httpx.HTTPError, Exception) as e:
                if attempt == 3:
                    failed += 1
                    if failed <= 10:
                        print(f"  Error: {e}")
                    return "", 0, 0
                await asyncio.sleep(2 ** attempt)

    return "", 0, 0


# ── Tier-specific R1 system prompts ──

TIER_PROMPTS = {
    "entities": """\
Generate a test case for evaluating an activation oracle's ability to name specific actors/entities.

Generate:
1. A 10-sentence text about a specific topic with at least 4-5 named entities (people, organizations, locations, species, compounds) playing distinct roles
2. An answerable question whose correct answer requires naming specific entities from the text
3. An unanswerable question about a plausible but ABSENT entity — something that sounds like it could be in the text but isn't mentioned at all
4. A brief description of the absent detail

The unanswerable question should be specific enough that someone who read the text would clearly know the asked-about entity is not mentioned.
Return ONLY valid JSON with keys: "text", "query", "absent_query", "domain", "absent_detail"
Do NOT wrap in markdown code blocks.""",

    "quantities": """\
Generate a test case for evaluating an activation oracle's ability to cite precise numbers.

Generate:
1. A 10-sentence text containing at least 5-6 precise quantities (percentages, measurements, dates, counts, monetary values, ratios) central to the narrative
2. An answerable question whose correct answer requires citing specific numbers from the text
3. An unanswerable question about a plausible but ABSENT metric/measurement — something that sounds like it could be in the text but isn't mentioned at all
4. A brief description of the absent detail

The unanswerable question should ask about a specific type of measurement that is clearly not present in the text.
Return ONLY valid JSON with keys: "text", "query", "absent_query", "domain", "absent_detail"
Do NOT wrap in markdown code blocks.""",

    "relationships": """\
Generate a test case for evaluating an activation oracle's ability to describe causal chains and relationships.

Generate:
1. A 10-sentence text describing specific causal chains, dependencies, or interactions (A leads to B which causes C; X depends on Y but not Z)
2. An answerable question whose correct answer requires describing a specific causal/dependency relationship from the text
3. An unanswerable question about a plausible but ABSENT mechanism/relationship — something that sounds like it could be in the text but isn't mentioned
4. A brief description of the absent detail

The unanswerable question should ask about a specific mechanism or causal link that is not described in the text.
Return ONLY valid JSON with keys: "text", "query", "absent_query", "domain", "absent_detail"
Do NOT wrap in markdown code blocks.""",

    "multi_detail": """\
Generate a test case for evaluating an activation oracle's ability to combine multiple types of specific detail.

Generate:
1. A 10-sentence text combining named entities, precise quantities, specific dates, AND causal relationships into a coherent narrative
2. An answerable question whose correct answer requires synthesizing at least 3 types of detail (who + how much + when + why)
3. An unanswerable question about a plausible but ABSENT aspect — combining detail types that sound like they could be in the text but aren't
4. A brief description of the absent detail

The unanswerable question should ask about a specific multi-faceted detail that is not present.
Return ONLY valid JSON with keys: "text", "query", "absent_query", "domain", "absent_detail"
Do NOT wrap in markdown code blocks.""",
}


R2_ANSWERABLE_SYSTEM = """\
You are given a text and a question about that text. Answer the question specifically and accurately based on the text. \
Include concrete details from the text (numbers, names, relationships) in your answer. 2-4 sentences."""

R2_UNANSWERABLE_SYSTEM = """\
You are given a text and a question about something NOT present in the text. \
Describe what IS discussed in the text that's related to the question's topic, then clearly note that \
the specific detail asked about is absent from the text. 2-4 sentences."""


# ── Diverse topic seeds (per tier) ──

TIER_TOPICS = {
    "entities": [
        "a medical research team publishing conflicting findings", "a corporate merger between tech companies",
        "archaeological excavations at multiple historical sites", "a gene therapy trial involving specific patients",
        "international trade negotiations between named countries", "a legal case involving multiple named parties",
        "species interactions in a named national park", "a collaboration between named universities on particle physics",
        "competing pharmaceutical companies developing similar drugs", "three rival architects designing a cathedral",
        "multiple volcanic eruptions in the Pacific Ring of Fire", "rival expeditions to Antarctica",
        "parallel discoveries in quantum mechanics", "competing species in a coral reef ecosystem",
    ],
    "quantities": [
        "clinical trial results for a specific drug dosage", "quarterly earnings breakdown by product line",
        "seismic wave measurements from a specific earthquake", "chemical reaction yields at precise temperatures",
        "vote counts in a local election by district", "battery discharge curves at different temperatures",
        "nutrient concentrations in soil samples by depth", "fuel efficiency data for specific aircraft models",
        "astronomical distance measurements to nearby stars", "demographic census data for a small country",
        "earthquake magnitude and damage statistics", "atmospheric CO2 concentration measurements over decades",
        "clinical trial dosage-response curves", "battery capacity degradation rates",
    ],
    "relationships": [
        "enzyme cascade triggering cellular apoptosis", "how interest rate changes propagate through an economy",
        "gene regulatory network controlling cell differentiation", "how deforestation affects local water cycles",
        "cascading failures in a power grid", "how ocean currents influence weather patterns",
        "how antibiotic resistance spreads through bacterial populations", "how soil chemistry affects wine characteristics",
        "feedback loops in climate systems", "dose-response relationships in pharmacology",
        "predator-prey population dynamics", "supply and demand effects on commodity prices",
        "positive and negative feedback in electronic circuits", "nutrient cycling in forest ecosystems",
    ],
    "multi_detail": [
        "a clinical trial with named drugs, specific patient counts, dates, and outcomes",
        "a corporate acquisition with named companies, dollar amounts, dates, and market effects",
        "a space mission with named crew, specific orbital parameters, dates, and scientific findings",
        "an ecological disaster with named location, measured contamination levels, timeline, and species impacts",
        "a historical battle with named commanders, troop counts, dates, and strategic consequences",
        "a construction project with named firms, material specifications, deadlines, and structural outcomes",
        "a scientific experiment with named researchers, precise measurements, dates, and causal conclusions",
        "a financial crisis with named institutions, specific losses, timeline, and regulatory responses",
    ],
}


# ── Phase runner ──

async def run_phase(
    tasks: list[tuple[str, str]],
    api_key: str,
    max_budget: float,
    max_tokens: int = 800,
) -> list[str]:
    """Run a batch of API calls. Returns list of response strings (empty on failure)."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY + 10, max_keepalive_connections=CONCURRENCY)

    async with httpx.AsyncClient(limits=limits) as client:
        coros = [
            call_openrouter(client, sys_p, user_p, api_key, semaphore, max_budget, max_tokens)
            for sys_p, user_p in tasks
        ]
        results = await asyncio.gather(*coros)

    return [r[0] for r in results]


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


# ── Main pipeline ──

def main():
    global budget_exceeded, completed, failed, total_input_tokens, total_output_tokens

    parser = argparse.ArgumentParser(description="Generate unified CoT description eval dataset with calibration")
    parser.add_argument("--output-dir", default="data/cot_description")
    parser.add_argument("--n", type=int, default=1000, help="Total items (split across 4 tiers, each produces answerable + unanswerable)")
    parser.add_argument("--max-budget", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", choices=["r1"], default=None, help="Resume from Round 1 JSONL")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after generation")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts and exit")
    args = parser.parse_args()

    api_key = os.environ["OPENROUTER_API_KEY"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    t0 = time.time()

    tiers = list(TIER_PROMPTS.keys())
    # Each text yields 2 items (answerable + unanswerable), so we need n/2 texts total
    n_texts = args.n // 2
    n_per_tier = n_texts // len(tiers)
    remainder = n_texts % len(tiers)

    print(f"\n{'='*60}")
    print(f"  Generating cot_description dataset (~{args.n} items from ~{n_texts} texts)")
    print(f"  Tiers: {tiers}")
    print(f"  ~{n_per_tier} texts per tier (+{remainder} extra)")
    print(f"  Model: {MODEL_ID}")
    print(f"{'='*60}")

    r1_file = output_dir / "cot_description_r1.jsonl"
    final_file = output_dir / "cot_description.jsonl"

    # ══════════════════════════════════════════════════════════════
    # Round 1: Generate texts + query pairs (answerable + unanswerable)
    # ══════════════════════════════════════════════════════════════

    if args.resume_from == "r1":
        print(f"\nResuming from {r1_file}")
        r1_rows = []
        with open(r1_file) as f:
            for line in f:
                if line.strip():
                    r1_rows.append(json.loads(line))
        print(f"  Loaded {len(r1_rows)} rows")
    else:
        r1_task_list = []  # (system, user, tier)
        for tier_idx, tier in enumerate(tiers):
            n_this_tier = n_per_tier + (1 if tier_idx < remainder else 0)
            system_prompt = TIER_PROMPTS[tier]
            topics = TIER_TOPICS[tier]

            for i in range(n_this_tier):
                topic = topics[i % len(topics)]
                user_prompt = f"Topic suggestion (use as inspiration, vary the specifics): {topic}"
                r1_task_list.append((system_prompt, user_prompt, tier))

        rng.shuffle(r1_task_list)

        if args.dry_run:
            for tier in tiers:
                print(f"\n[DRY RUN] Tier: {tier}")
                print(f"  System prompt:\n{TIER_PROMPTS[tier][:300]}...")
                sample_topic = TIER_TOPICS[tier][0]
                print(f"  Sample user prompt: Topic suggestion (use as inspiration, vary the specifics): {sample_topic}")
            print(f"\n  Would make {len(r1_task_list)} R1 calls + ~{len(r1_task_list)*2} R2 calls")
            return

        print(f"\nRound 1: Generate text + query pairs ({len(r1_task_list)} calls)")
        r1_api_tasks = [(sys_p, user_p) for sys_p, user_p, _ in r1_task_list]
        r1_responses = asyncio.run(run_phase(r1_api_tasks, api_key, args.max_budget, max_tokens=700))

        r1_rows = []
        r1_failures = 0
        for resp, (_, _, tier) in zip(r1_responses, r1_task_list):
            if not resp:
                r1_failures += 1
                continue
            parsed = _parse_json_response(resp)
            if parsed is None or "text" not in parsed or "query" not in parsed or "absent_query" not in parsed:
                r1_failures += 1
                continue
            parsed["tier"] = tier
            r1_rows.append(parsed)

        print(f"  Round 1 done: {len(r1_rows)} ok, {r1_failures} failures")
        tier_counts = {}
        for row in r1_rows:
            tier_counts[row["tier"]] = tier_counts.get(row["tier"], 0) + 1
        for tier, count in sorted(tier_counts.items()):
            print(f"    {tier}: {count}")

        with open(r1_file, "w") as f:
            for row in r1_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(r1_rows)} rows -> {r1_file}")

    if budget_exceeded:
        print("Budget exceeded after Round 1. Partial results saved.")
        _print_summary(r1_rows, t0)
        return

    # ══════════════════════════════════════════════════════════════
    # Round 2: Generate reference answers (answerable + unanswerable)
    # ══════════════════════════════════════════════════════════════

    print(f"\nRound 2: Generate reference answers ({len(r1_rows) * 2} calls)")

    r2_tasks = []
    r2_meta = []  # track (row_idx, answerable)
    for i, row in enumerate(r1_rows):
        # Answerable
        r2_tasks.append((R2_ANSWERABLE_SYSTEM, f"## Text\n{row['text']}\n\n## Question\n{row['query']}"))
        r2_meta.append((i, True))
        # Unanswerable
        r2_tasks.append((R2_UNANSWERABLE_SYSTEM, f"## Text\n{row['text']}\n\n## Question\n{row['absent_query']}"))
        r2_meta.append((i, False))

    r2_responses = asyncio.run(run_phase(r2_tasks, api_key, args.max_budget, max_tokens=300))

    final_rows = []
    r2_failures = 0
    for j, resp in enumerate(r2_responses):
        if not resp:
            r2_failures += 1
            continue
        row_idx, answerable = r2_meta[j]
        row = r1_rows[row_idx]
        query = row["query"] if answerable else row["absent_query"]
        final_rows.append({
            "cot_text": row["text"],
            "prompt": query,
            "target_response": resp.strip(),
            "tier": row["tier"],
            "domain": row.get("domain", ""),
            "answerable": answerable,
            "absent_detail": row.get("absent_detail", "") if not answerable else "",
            "task": "cot_description",
        })

    print(f"  Round 2 done: {len(final_rows)} ok, {r2_failures} failures")

    # Prune to target size (~250 per tier, balanced answerable/unanswerable)
    target_per_tier = args.n // len(tiers)
    pruned_rows = _prune_balanced(final_rows, tiers, target_per_tier, rng)

    with open(final_file, "w") as f:
        for row in pruned_rows:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved {len(pruned_rows)} rows -> {final_file}")

    _print_summary(pruned_rows, t0)

    # ══════════════════════════════════════════════════════════════
    # Upload to HuggingFace
    # ══════════════════════════════════════════════════════════════

    if args.upload and pruned_rows:
        _upload_to_hf(final_file, pruned_rows)


def _prune_balanced(rows: list[dict], tiers: list[str], target_per_tier: int, rng: random.Random) -> list[dict]:
    """Prune to ~target_per_tier items per tier, balanced 50/50 answerable/unanswerable."""
    result = []
    for tier in tiers:
        tier_ans = [r for r in rows if r["tier"] == tier and r["answerable"]]
        tier_unans = [r for r in rows if r["tier"] == tier and not r["answerable"]]
        half = target_per_tier // 2
        if len(tier_ans) > half:
            rng.shuffle(tier_ans)
            tier_ans = tier_ans[:half]
        if len(tier_unans) > half:
            rng.shuffle(tier_unans)
            tier_unans = tier_unans[:half]
        result.extend(tier_ans)
        result.extend(tier_unans)
        print(f"    {tier}: {len(tier_ans)} answerable + {len(tier_unans)} unanswerable = {len(tier_ans) + len(tier_unans)}")
    rng.shuffle(result)
    return result


def _upload_to_hf(jsonl_path: Path, rows: list[dict]):
    """Upload dataset to HuggingFace as parquet."""
    import pandas as pd
    from huggingface_hub import HfApi

    repo_id = "mats-10-sprint-cs-jb/cot-oracle-eval-cot-description"
    print(f"\nUploading to {repo_id}...")

    df = pd.DataFrame(rows)
    # Ensure consistent types
    df["answerable"] = df["answerable"].astype(bool)
    col_order = ["cot_text", "prompt", "target_response", "tier", "domain", "answerable", "absent_detail", "task"]
    df = df[col_order]

    parquet_path = jsonl_path.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded {len(rows)} rows to {repo_id}")


def _print_summary(rows: list[dict], t0: float):
    elapsed = time.time() - t0
    cost = estimate_cost()

    print(f"\n{'='*60}")
    print(f"  COT_DESCRIPTION SUMMARY")
    print(f"  Total rows: {len(rows)}")
    print(f"  API calls: {completed} completed, {failed} failed")
    print(f"  Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out")
    print(f"  Cost: ${cost:.3f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Tier breakdown
    tier_counts = {}
    ans_counts = {"answerable": 0, "unanswerable": 0}
    for row in rows:
        t = row.get("tier", "unknown")
        tier_counts[t] = tier_counts.get(t, 0) + 1
        if row.get("answerable", True):
            ans_counts["answerable"] += 1
        else:
            ans_counts["unanswerable"] += 1
    print(f"  Tier breakdown: {dict(sorted(tier_counts.items()))}")
    print(f"  Answerable split: {ans_counts}")

    # Sample items
    n_show = min(4, len(rows))
    if n_show > 0 and "cot_text" in rows[0]:
        print(f"\n  Sample items:")
        for i in range(n_show):
            row = rows[i]
            text_short = row["cot_text"][:100].replace("\n", " ")
            print(f"    [{i}] tier={row.get('tier', '?')} answerable={row.get('answerable', '?')}")
            print(f"        text: {text_short}...")
            print(f"        query: {row['prompt']}")
            print(f"        answer: {row['target_response'][:100]}...")


if __name__ == "__main__":
    main()
