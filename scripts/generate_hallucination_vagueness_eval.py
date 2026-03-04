#!/usr/bin/env python3
"""
Generate synthetic hallucination & vagueness eval datasets with granularity tiers.

3-round pipeline using Gemini 2.5 Flash Lite via OpenRouter:
  Round 1: Design adversarial test cases (text + query) per tier
  Round 2: Generate ground truth reference answers
  Round 3: (at eval time — handled by eval_loop.py, not this script)

Hallucination tiers: domain_confusion, entity_swap, quantity_fabrication, subtle_distortion
Vagueness tiers:     named_entities, exact_quantities, relationships, multi_detail

Usage:
    OPENROUTER_API_KEY=sk-... python scripts/generate_hallucination_vagueness_eval.py
    python scripts/generate_hallucination_vagueness_eval.py --n 100 --max-budget 2.0 --dry-run
    python scripts/generate_hallucination_vagueness_eval.py --n 100 --max-budget 2.0 --upload
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

MODEL_ID = "google/gemini-2.5-flash-lite"
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

HALLUCINATION_TIER_PROMPTS = {
    "domain_confusion": """\
Design a test case to detect DOMAIN CONFUSION hallucination in an activation oracle.

The oracle reads activations from a model processing text about Topic A, but hallucinates by describing Topic B (a superficially similar but different domain).

Generate:
1. A 10-sentence text on a specific technical topic with concrete details (numbers, names, formulas, dates)
2. An open-ended question about the text where a hallucinating oracle would confuse the domain entirely — e.g., mistaking chemistry for biology, astronomy for geology, one historical period for another

The text and question should be designed so that domain confusion is clearly detectable (the correct domain and the confused domain should be obviously different).
Return ONLY valid JSON with keys: "text", "query", "domain", "trap_description"
Do NOT wrap in markdown code blocks.""",

    "entity_swap": """\
Design a test case to detect ENTITY SWAP hallucination in an activation oracle.

The oracle reads activations from a model processing text with multiple named entities, but hallucinates by swapping or confusing which entity did what.

Generate:
1. A 10-sentence text mentioning at least 3-4 distinct named entities (people, organizations, locations, chemical compounds, species) with specific actions or properties attributed to each
2. An open-ended question where a hallucinating oracle would attribute the wrong entity's actions/properties to a different entity

The entities should be distinct enough that swapping them is clearly wrong, but similar enough that a confused oracle might mix them up.
Return ONLY valid JSON with keys: "text", "query", "domain", "trap_description"
Do NOT wrap in markdown code blocks.""",

    "quantity_fabrication": """\
Design a test case to detect QUANTITY FABRICATION hallucination in an activation oracle.

The oracle reads activations from a model processing text with precise numerical data, but hallucinates by inventing plausible-sounding but wrong numbers.

Generate:
1. A 10-sentence text containing at least 5 specific quantities (measurements, percentages, dates, counts, ratios, temperatures, distances) that are central to the meaning
2. An open-ended question whose correct answer requires citing specific numbers from the text — a hallucinating oracle would fabricate nearby but wrong values

The numbers should be specific enough that fabrication is detectable (e.g., "47.3%" not "about half").
Return ONLY valid JSON with keys: "text", "query", "domain", "trap_description"
Do NOT wrap in markdown code blocks.""",

    "subtle_distortion": """\
Design a test case to detect SUBTLE DISTORTION hallucination in an activation oracle.

The oracle reads activations from a model processing text with precise causal/logical relationships, but hallucinates by subtly distorting the relationship (reversing cause and effect, changing direction of a trend, flipping a comparison).

Generate:
1. A 10-sentence text with clear directional relationships (A causes B, X is greater than Y, temperature increases lead to Z decreasing, etc.)
2. An open-ended question where a hallucinating oracle would get the direction or causality backwards — the answer would sound plausible but have the relationship reversed

The relationships should be specific enough that reversal is clearly wrong, not ambiguous.
Return ONLY valid JSON with keys: "text", "query", "domain", "trap_description"
Do NOT wrap in markdown code blocks.""",
}

VAGUENESS_TIER_PROMPTS = {
    "named_entities": """\
Design a test case to detect VAGUENESS about named entities in an activation oracle's response.

Generate:
1. A 10-sentence text mentioning at least 4-5 specific named entities (people, organizations, locations, species, chemical compounds) with distinct roles
2. An open-ended question whose correct answer REQUIRES naming those specific entities — a vague answer like "the model is processing information about various entities" or "several organizations are involved" would be clearly inadequate

The question should be impossible to answer well without using the actual names from the text.
Return ONLY valid JSON with keys: "text", "query", "domain", "specifics_required"
Do NOT wrap in markdown code blocks.""",

    "exact_quantities": """\
Design a test case to detect VAGUENESS about numerical quantities in an activation oracle's response.

Generate:
1. A 10-sentence text containing at least 5-6 precise quantities (exact percentages, measurements, dates, counts, monetary values, ratios) that are central to the narrative
2. An open-ended question whose correct answer REQUIRES citing the specific numbers — a vague answer like "the model is working with numerical calculations" or "various measurements were taken" would be clearly inadequate

The question should be impossible to answer well without mentioning the actual numerical values.
Return ONLY valid JSON with keys: "text", "query", "domain", "specifics_required"
Do NOT wrap in markdown code blocks.""",

    "relationships": """\
Design a test case to detect VAGUENESS about relationships and causation in an activation oracle's response.

Generate:
1. A 10-sentence text describing specific causal chains, dependencies, or interactions between entities (A leads to B which causes C; X depends on Y but not Z; increasing W decreases V)
2. An open-ended question whose correct answer REQUIRES describing the specific causal/dependency relationships — a vague answer like "the model is analyzing how things are connected" or "there are complex interactions" would be clearly inadequate

The question should be impossible to answer well without describing the actual mechanism or causal chain.
Return ONLY valid JSON with keys: "text", "query", "domain", "specifics_required"
Do NOT wrap in markdown code blocks.""",

    "multi_detail": """\
Design a test case to detect VAGUENESS when multiple specific details are needed simultaneously.

Generate:
1. A 10-sentence text that combines named entities, precise quantities, specific dates, AND causal relationships into a coherent narrative
2. An open-ended question whose correct answer REQUIRES combining multiple types of specific detail (who + how much + when + why) — a vague answer touching on only one dimension or speaking in generalities would be clearly inadequate

The question should require synthesizing at least 3 different types of specific detail from the text.
Return ONLY valid JSON with keys: "text", "query", "domain", "specifics_required"
Do NOT wrap in markdown code blocks.""",
}

R2_SYSTEM = """\
You are given a text and a question about that text. Answer the question specifically and accurately based on the text. \
Include concrete details from the text (numbers, names, relationships) in your answer. 2-4 sentences."""

# ── Diverse topic seeds (per tier) ──

HALLUCINATION_TOPICS = {
    "domain_confusion": [
        "enzyme kinetics in liver metabolism", "orbital mechanics of asteroid mining",
        "tidal patterns in fjord systems", "semiconductor doping concentrations",
        "population genetics of island finch species", "hydraulic engineering of Roman aqueducts",
        "crystallography of mineral formations", "thermodynamic cycles in jet engines",
    ],
    "entity_swap": [
        "competing pharmaceutical companies developing similar drugs", "three rival architects designing a cathedral",
        "multiple volcanic eruptions in the Pacific Ring of Fire", "four competing protocols in early internet history",
        "rival expeditions to Antarctica", "parallel discoveries in quantum mechanics",
        "competing species in a coral reef ecosystem", "multiple rivers converging in a delta system",
    ],
    "quantity_fabrication": [
        "clinical trial dosage-response curves", "quarterly financial performance metrics",
        "astronomical distance measurements to nearby stars", "chemical reaction yield percentages",
        "demographic census data for a small country", "earthquake magnitude and damage statistics",
        "atmospheric CO2 concentration measurements over decades", "battery capacity degradation rates",
    ],
    "subtle_distortion": [
        "feedback loops in climate systems", "dose-response relationships in pharmacology",
        "predator-prey population dynamics", "supply and demand effects on commodity prices",
        "positive and negative feedback in electronic circuits", "nutrient cycling in forest ecosystems",
        "cause and effect in epidemiological studies", "correlation vs causation in social science research",
    ],
}

VAGUENESS_TOPICS = {
    "named_entities": [
        "a medical research team publishing conflicting findings", "a corporate merger between tech companies",
        "archaeological excavations at multiple historical sites", "a gene therapy trial involving specific patients",
        "international trade negotiations between named countries", "a legal case involving multiple named parties",
        "species interactions in a named national park", "a collaboration between named universities on particle physics",
    ],
    "exact_quantities": [
        "clinical trial results for a specific drug dosage", "quarterly earnings breakdown by product line",
        "seismic wave measurements from a specific earthquake", "chemical reaction yields at precise temperatures",
        "vote counts in a local election by district", "battery discharge curves at different temperatures",
        "nutrient concentrations in soil samples by depth", "fuel efficiency data for specific aircraft models",
    ],
    "relationships": [
        "enzyme cascade triggering cellular apoptosis", "how interest rate changes propagate through an economy",
        "gene regulatory network controlling cell differentiation", "how deforestation affects local water cycles",
        "cascading failures in a power grid", "how ocean currents influence weather patterns",
        "how antibiotic resistance spreads through bacterial populations", "how soil chemistry affects wine characteristics",
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

    parser = argparse.ArgumentParser(description="Generate hallucination & vagueness eval datasets with granularity tiers")
    parser.add_argument("--output-dir", default="data/hallucination_vagueness")
    parser.add_argument("--n", type=int, default=100, help="Total items per dataset (split across tiers)")
    parser.add_argument("--max-budget", type=float, default=2.0)
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

    eval_configs = {
        "hallucination": {
            "tier_prompts": HALLUCINATION_TIER_PROMPTS,
            "tier_topics": HALLUCINATION_TOPICS,
        },
        "vagueness": {
            "tier_prompts": VAGUENESS_TIER_PROMPTS,
            "tier_topics": VAGUENESS_TOPICS,
        },
    }

    for eval_type, cfg in eval_configs.items():
        tiers = list(cfg["tier_prompts"].keys())
        n_per_tier = args.n // len(tiers)
        remainder = args.n % len(tiers)

        print(f"\n{'='*60}")
        print(f"  Generating {eval_type} dataset ({args.n} items, {len(tiers)} tiers)")
        print(f"  Tiers: {tiers}")
        print(f"  ~{n_per_tier} items per tier (+{remainder} extra)")
        print(f"{'='*60}")

        # Reset stats per dataset
        completed = 0
        failed = 0
        total_input_tokens = 0
        total_output_tokens = 0
        budget_exceeded = False

        r1_file = output_dir / f"{eval_type}_r1.jsonl"
        final_file = output_dir / f"{eval_type}.jsonl"

        # ══════════════════════════════════════════════════════════════
        # Round 1: Design adversarial test cases (per tier)
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
            # Build tier-aware tasks: (system_prompt, user_prompt, tier_name)
            r1_task_list = []  # (system, user, tier)
            for tier_idx, tier in enumerate(tiers):
                n_this_tier = n_per_tier + (1 if tier_idx < remainder else 0)
                system_prompt = cfg["tier_prompts"][tier]
                topics = cfg["tier_topics"][tier]

                for i in range(n_this_tier):
                    topic = topics[i % len(topics)]
                    user_prompt = f"Topic suggestion (use as inspiration, vary the specifics): {topic}"
                    r1_task_list.append((system_prompt, user_prompt, tier))

            rng.shuffle(r1_task_list)

            if args.dry_run:
                for tier in tiers:
                    print(f"\n[DRY RUN] Tier: {tier}")
                    print(f"  System prompt:\n{cfg['tier_prompts'][tier][:200]}...")
                continue

            print(f"\nRound 1: Generate test cases ({len(r1_task_list)} calls)")
            r1_api_tasks = [(sys_p, user_p) for sys_p, user_p, _ in r1_task_list]
            r1_responses = asyncio.run(run_phase(r1_api_tasks, api_key, args.max_budget, max_tokens=600))

            r1_rows = []
            r1_failures = 0
            for resp, (_, _, tier) in zip(r1_responses, r1_task_list):
                if not resp:
                    r1_failures += 1
                    continue
                parsed = _parse_json_response(resp)
                if parsed is None or "text" not in parsed or "query" not in parsed:
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
            _print_summary(eval_type, r1_rows, t0)
            continue

        # ══════════════════════════════════════════════════════════════
        # Round 2: Generate ground truth reference answers
        # ══════════════════════════════════════════════════════════════

        print(f"\nRound 2: Generate reference answers ({len(r1_rows)} calls)")

        r2_tasks = [
            (R2_SYSTEM, f"## Text\n{row['text']}\n\n## Question\n{row['query']}")
            for row in r1_rows
        ]
        r2_responses = asyncio.run(run_phase(r2_tasks, api_key, args.max_budget, max_tokens=300))

        final_rows = []
        r2_failures = 0
        for i, resp in enumerate(r2_responses):
            if not resp:
                r2_failures += 1
                continue
            row = r1_rows[i]
            final_rows.append({
                "cot_text": row["text"],
                "prompt": row["query"],
                "target_response": resp.strip(),
                "tier": row["tier"],
                "domain": row.get("domain", ""),
                "trap_description": row.get("trap_description", ""),
                "specifics_required": row.get("specifics_required", ""),
                "task": eval_type,
            })

        print(f"  Round 2 done: {len(final_rows)} ok, {r2_failures} failures")

        with open(final_file, "w") as f:
            for row in final_rows:
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(final_rows)} rows -> {final_file}")

        _print_summary(eval_type, final_rows, t0)

        # ══════════════════════════════════════════════════════════════
        # Upload to HuggingFace
        # ══════════════════════════════════════════════════════════════

        if args.upload and final_rows:
            _upload_to_hf(eval_type, final_file, final_rows)


def _upload_to_hf(eval_type: str, jsonl_path: Path, rows: list[dict]):
    """Upload dataset to HuggingFace as parquet."""
    import pandas as pd
    from huggingface_hub import HfApi

    repo_id = f"mats-10-sprint-cs-jb/cot-oracle-{eval_type}"
    print(f"\nUploading to {repo_id}...")

    df = pd.DataFrame(rows)
    # Flatten any list/bool-valued cells to strings (Gemini sometimes returns lists or booleans)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else str(x) if not isinstance(x, str) else x)
    # Column order: cot_text, prompt, target_response, tier, metadata..., task last
    col_order = ["cot_text", "prompt", "target_response", "tier", "domain"]
    col_order += [c for c in df.columns if c not in col_order and c != "task"]
    col_order.append("task")
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


def _print_summary(eval_type: str, rows: list[dict], t0: float):
    elapsed = time.time() - t0
    cost = estimate_cost()

    print(f"\n{'='*60}")
    print(f"  {eval_type.upper()} SUMMARY")
    print(f"  Total rows: {len(rows)}")
    print(f"  API calls: {completed} completed, {failed} failed")
    print(f"  Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out")
    print(f"  Cost: ${cost:.3f}")
    print(f"  Elapsed: {elapsed:.0f}s")

    # Show tier breakdown
    tier_counts = {}
    for row in rows:
        t = row.get("tier", "unknown")
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"  Tier breakdown: {dict(sorted(tier_counts.items()))}")

    # Show a few examples
    n_show = min(3, len(rows))
    if n_show > 0 and "cot_text" in rows[0]:
        print(f"\n  Sample items:")
        for i in range(n_show):
            row = rows[i]
            text_short = row["cot_text"][:100].replace("\n", " ")
            print(f"    [{i}] tier={row.get('tier', '?')} text: {text_short}...")
            print(f"        query: {row['prompt']}")
            print(f"        answer: {row['target_response'][:100]}...")


if __name__ == "__main__":
    main()
