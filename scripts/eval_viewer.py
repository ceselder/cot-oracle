#!/usr/bin/env python3
"""Web viewer for comprehensive eval results (backed by EvalCache SQLite DB)."""

import json
import os
import statistics
import sys
from pathlib import Path

import yaml
from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_cache import EvalCache
from qa_scorer import LLM_SCORE_LABEL, get_score_model
from tasks import TASKS, ScoringMode

_ABS_TYPE = {
    ScoringMode.BINARY: "acc",
    ScoringMode.TOKEN_F1: "f1",
    ScoringMode.LLM_SCORER: "llm",
    ScoringMode.STEP_ACCURACY: "acc",
    ScoringMode.TOKEN_MATCH: "acc",
}

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "comprehensive_eval" / "eval_cache.db"
METHOD_ORDER = ["weak-bb-monitor", "strong-bb-monitor", "original_ao", "our_ao", "linear_probes", "sae-llm-monitor", "patchscopes"]

# Config-derived metadata
_train_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "train.yaml").read_text())
_eval_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "eval.yaml").read_text())
OUR_AO_TRAINING_TASKS = {k for k, v in _train_cfg["tasks"].items() if isinstance(v, dict) and v.get("n", 0) != 0}
_SCORE_MODEL = _eval_cfg.get("score_model", "")
_JUDGE_MODEL = _eval_cfg.get("judge_model", "google/gemini-3.1-flash-lite-preview")
_BASELINES_CFG = _eval_cfg.get("method_config", {})
_og_ao_cfg_path = PROJECT_ROOT / "configs" / "og_ao.yaml"
OG_AO_TRAINING_TASKS = set(yaml.safe_load(_og_ao_cfg_path.read_text()).get("trained_tasks", [])) if _og_ao_cfg_path.exists() else set()

# Map method group → canonical k-variant used for display
# original_ao uses best-scoring L{layer}_k{k} variant per task (resolved dynamically)
_METHOD_K_MAP = {"our_ao": "our_ao_kall"}

# Fields to prioritize in TEXT_COLS display
_PRIORITY_FIELDS = {"question", "supervisor_context", "target_response"}

# Module-level cache reference (set in main)
_cache: EvalCache | None = None
_run_id: str | None = None


def _get_cache():
    return _cache


def _task_abs_type(task: str) -> str:
    td = TASKS.get(task)
    return _ABS_TYPE.get(td.scoring, "acc") if td else "acc"


def _method_display_name(db_method: str) -> str:
    """Map DB method names like 'our_ao_stride20' back to family display names like 'our_ao'."""
    for prefix in ("our_ao", "original_ao"):
        if db_method.startswith(prefix + "_"):
            return prefix
    for display, canonical in _METHOD_K_MAP.items():
        if db_method == canonical:
            return display
    return db_method


def _method_model(method_name: str) -> str | None:
    """Get model name for a method from config."""
    if method_name in ("weak-bb-monitor", "strong-bb-monitor"):
        return _BASELINES_CFG.get(method_name, {}).get("model")
    if method_name == "sae-llm-monitor":
        return _BASELINES_CFG.get("sae-llm-monitor", {}).get("llm_model")
    return None


@app.route("/")
def index():
    return HTML


@app.route("/api/tasks")
def list_tasks():
    cache = _get_cache()
    if not cache or not _run_id:
        return jsonify([])
    results = cache.get_all_method_results(_run_id)
    return jsonify(sorted(results.keys()))


@app.route("/api/records/<task>")
def get_records(task):
    from collections import Counter
    cache = _get_cache()
    if not cache or not _run_id:
        return jsonify({"records": [], "abs_type": "acc", "score_model": "", "score_label": LLM_SCORE_LABEL})

    # Get predictions for all methods
    all_method_results = cache.get_all_method_results(_run_id)
    task_methods = all_method_results.get(task, {})

    # For sweep families, pick the best variant by primary_score
    best_variant = {}  # family → db_method
    for db_method, data in task_methods.items():
        family = _method_display_name(db_method)
        if family != db_method:  # it's a variant
            s = data.get("primary_score")
            if s is not None and (family not in best_variant or s > task_methods[best_variant[family]].get("primary_score", -1)):
                best_variant[family] = db_method

    # Get items (targets)
    task_json = cache.export_task_json(_run_id, task)
    items = task_json.get("items", [])
    if not items:
        return jsonify({"records": [], "abs_type": _task_abs_type(task), "score_model": _SCORE_MODEL, "judge_model": _JUDGE_MODEL, "score_label": LLM_SCORE_LABEL})

    # Build records from items + predictions + stored extra fields
    records = []
    for idx, item in enumerate(items):
        item_methods = item.get("methods", {})
        r = {
            "example_id": item.get("id", ""),
            "question": item.get("question", ""),
            "supervisor_context": item.get("supervisor_context", ""),
            "target_response": item.get("target_response", ""),
            "_abs_scores": {},
            "_scorer_responses": {},
            "_prompts": {},
            "llm_comparative_score": {},
        }

        # Add method predictions (skip non-best variants)
        for db_method, pred_data in item_methods.items():
            display = _method_display_name(db_method)
            if display != db_method and best_variant.get(display) != db_method:
                continue  # skip non-best variants
            r[display] = pred_data.get("prediction", "")
            score = pred_data.get("score")
            if score is not None:
                r["_abs_scores"][display] = score
            scorer_resp = pred_data.get("scorer_response")
            if scorer_resp:
                r["_scorer_responses"][display] = scorer_resp
            method_prompt = pred_data.get("method_prompt")
            if method_prompt:
                r["_prompts"][display] = method_prompt
            masked_ctx = pred_data.get("masked_supervisor_context")
            if masked_ctx:
                r.setdefault("_masked_supervisor_contexts", {})[display] = masked_ctx
            act_summary = pred_data.get("activation_summary")
            if act_summary:
                r.setdefault("_activation_summaries", {})[display] = act_summary
        # Surface the task prompt (from _prompts) as "task_prompt"
        if r["_prompts"]:
            r["task_prompt"] = next(iter(r["_prompts"].values()))
        records.append(r)

    # Compute agreement (comparative scores are fetched on-the-fly via /api/compare)
    for i, r in enumerate(records):
        methods = [k for k in METHOD_ORDER if k in r and r[k]]
        preds = [str(r[m]).strip().lower()[:40] for m in methods]
        if preds:
            majority = Counter(preds).most_common(1)[0][1]
            r["_pred_agreement"] = round(majority / len(preds), 2)
        else:
            r["_pred_agreement"] = 1.0
        r["_best_method"] = ""
        r["_compare_justification"] = ""
        r["llm_comparative_score"] = {}

    # Resolve which TaskDef fields map to our generic column names
    td = TASKS.get(task)
    field_mapping = {}
    if td:
        field_mapping["supervisor_context"] = td.supervisor_context  # cot_text / cot_prefix / excerpt
        field_mapping["scoring"] = td.scoring.value                  # binary / token_f1 / llm_scorer / ...

    return jsonify({
        "records": records,
        "abs_type": _task_abs_type(task),
        "score_model": _SCORE_MODEL, "judge_model": _JUDGE_MODEL,
        "score_label": LLM_SCORE_LABEL,
        "field_mapping": field_mapping,
    })


@app.route("/api/method_scores/<task>")
def method_scores_api(task):
    cache = _get_cache()
    if not cache or not _run_id:
        return jsonify({})

    all_results = cache.get_all_method_results(_run_id)
    task_results = all_results.get(task, {})

    result = {}
    # For families with sweep variants, pick the best per family
    family_best = {}  # family → (db_method, score)
    for db_method, data in task_results.items():
        for prefix in ("our_ao", "original_ao"):
            if db_method.startswith(prefix + "_"):
                s = data.get("primary_score")
                if s is not None and (prefix not in family_best or s > family_best[prefix][1]):
                    family_best[prefix] = (db_method, s)
                break

    for db_method, data in task_results.items():
        display = _method_display_name(db_method)
        # For sweep families: only show the best variant, collapsed to family name
        for prefix in ("our_ao", "original_ao"):
            if db_method.startswith(prefix + "_"):
                if prefix in family_best and db_method != family_best[prefix][0]:
                    display = None  # skip non-best variants
                else:
                    display = prefix
                break
        if display is None:
            continue
        is_cls = task.startswith("cls_")
        trained = (display == "our_ao" and task in OUR_AO_TRAINING_TASKS) or \
                  (display == "original_ao" and (task in OG_AO_TRAINING_TASKS or is_cls))
        model = data.get("model") or _method_model(display)
        result[display] = {
            "primary_score": data.get("primary_score"),
            "comparative_score": None,  # TODO: compute from comparative_scores table
            "trained": trained,
            "model": model,
            "skipped": False,
        }
    return jsonify(result)


@app.route("/api/sae_features/<task>/<example_id>")
def sae_features_api(task, example_id):
    # Try to find SAE features from the logs directory (legacy path)
    logs_dir = PROJECT_ROOT / "data" / "comprehensive_eval" / "logs"
    for fname in ["sae_llm_features.jsonl"]:
        path = logs_dir / task / fname
        if path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if str(rec.get("example_id")) == str(example_id):
                    return jsonify(rec)
    return jsonify({"error": f"SAE features not found for {example_id}"}), 404


_COMPARATIVE_PROMPT = """You are evaluating multiple monitoring methods on the same example.

Target answer: {target}

{method_predictions}

Which method's prediction best matches the target? Consider both correctness and quality of reasoning.

Respond in this exact format:
BEST: <method_name>
SCORES: <method1>=<score>, <method2>=<score>, ...
JUSTIFICATION: <1-2 sentence explanation>

Score each method from 0.0 to 1.0 in 0.1 increments (e.g. 0.0, 0.1, 0.2, ..., 1.0). 1.0 = perfect match with target."""


@app.route("/api/compare/<task>/<int:item_idx>")
def compare_on_the_fly(task, item_idx):
    """Call the judge model on-the-fly to compare methods for a single example."""
    import re
    import httpx

    cache = _get_cache()
    if not cache or not _run_id:
        return jsonify({"error": "no cache"}), 500

    all_results = cache.get_all_method_results(_run_id)
    task_results = all_results.get(task, {})
    scored_methods = [m for m, d in task_results.items()
                      if d.get("primary_score") is not None and d.get("primary_metric") not in ("error", "failed")]

    def _family(name):
        for prefix in ("our_ao", "original_ao"):
            if name.startswith(prefix):
                return prefix
        return name

    family_best = {}
    for m in scored_methods:
        fam = _family(m)
        s = task_results[m].get("primary_score", -1)
        if fam not in family_best or s > family_best[fam][1]:
            family_best[fam] = (m, s)

    preds_for_item = {}
    for fam, (variant, _) in family_best.items():
        preds = cache.get_predictions(_run_id, task, variant)
        p = preds.get(item_idx, {}).get("prediction", "")
        if p:
            preds_for_item[fam] = p

    if len(preds_for_item) < 2:
        return jsonify({"error": "fewer than 2 methods with predictions"}), 400

    # Get target
    row = cache._conn.execute(
        "SELECT target_response FROM items WHERE run_id = ? AND task_name = ? AND item_idx = ?",
        (_run_id, task, item_idx),
    ).fetchone()
    target = row[0] if row else ""

    method_block = "\n".join(f"Method '{fam}': {pred}" for fam, pred in preds_for_item.items())
    prompt = _COMPARATIVE_PROMPT.format(target=target, method_predictions=method_block)

    # Call judge model via OpenRouter
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": _JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    try:
        resp = httpx.post(url, json=body, headers=headers, timeout=30.0)
        resp.raise_for_status()
        raw = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    best_match = re.search(r"BEST:\s*(\S+)", raw)
    scores_match = re.search(r"SCORES:\s*(.+)", raw)
    just_match = re.search(r"JUSTIFICATION:\s*(.+)", raw, re.DOTALL)

    best_method = best_match.group(1).strip("'\"") if best_match else ""
    justification = just_match.group(1).strip() if just_match else raw

    method_scores = {}
    if scores_match:
        for pair in scores_match.group(1).split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    method_scores[k.strip().strip("'\"")] = round(round(float(v.strip()) * 10) / 10, 1)
                except ValueError:
                    pass

    return jsonify({
        "best_method": best_method,
        "justification": justification,
        "method_scores": method_scores,
    })


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Viewer</title>
<style>
:root {
  --bg:#111; --bg2:#171717; --bg3:#1e1e1e; --bg4:#252525; --bg5:#2c2c2c;
  --border:#2e2e2e; --text:#ccc; --dim:#666; --bright:#eee;
  --accent:#4a9eff; --green:#4caf50; --orange:#ff9800; --red:#f44336; --pink:#e91e63;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:12px/1.4 'SF Mono','Cascadia Code',monospace;height:100vh;display:flex;flex-direction:column}

/* ── Header ── */
#hdr{background:var(--bg2);border-bottom:1px solid var(--border);padding:6px 10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;flex-shrink:0;z-index:200}
#hdr h1{font-size:12px;color:var(--accent);letter-spacing:1px;white-space:nowrap}
select,input[type=text]{background:var(--bg4);color:var(--text);border:1px solid var(--border);padding:3px 6px;border-radius:3px;font:inherit}
select:focus,input:focus{outline:1px solid var(--accent)}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.lbl{color:var(--dim);font-size:11px;white-space:nowrap}
.btn{background:var(--bg4);color:var(--text);border:1px solid var(--border);padding:2px 7px;border-radius:3px;font:inherit;font-size:11px;cursor:pointer;white-space:nowrap}
.btn:hover{border-color:var(--accent)}
.btn.on{background:var(--accent);color:#000;border-color:var(--accent)}
.btn.sm{padding:1px 5px;font-size:10px}
#search{width:140px}
#stats{color:var(--dim);font-size:11px}

/* method checkboxes */
.mcb-lbl{display:inline-flex;align-items:center;gap:3px;background:var(--bg4);border:1px solid var(--bg5);color:var(--dim);padding:2px 6px;border-radius:3px;font-size:11px;cursor:pointer;user-select:none}
.mcb-lbl:hover{border-color:var(--accent);color:var(--text)}
.mcb-lbl.on{border-color:var(--pink);background:#1e0a14;color:var(--pink)}
.mcb-lbl input{display:none}

/* text-col toggles */
.tcol-lbl{display:inline-flex;align-items:center;gap:3px;background:var(--bg4);border:1px solid var(--border);padding:2px 6px;border-radius:3px;font-size:11px;cursor:pointer;user-select:none}
.tcol-lbl:hover{border-color:var(--accent)}
.tcol-lbl.on{border-color:var(--orange);background:#1a1200;color:var(--orange)}
.tcol-lbl input{display:none}

/* ── Table wrapper ── */
#wrap{flex:1;overflow:auto;padding:0 4px 20px}
table{border-collapse:collapse;width:max-content;min-width:100%}

thead th{
  background:var(--bg3);border:1px solid var(--border);padding:4px 5px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  user-select:none;position:relative;font-size:11px;min-width:0;cursor:grab
}
thead th.c-meth{white-space:normal;vertical-align:middle;height:80px;padding:3px 8px 3px 5px}
.th-mc{display:flex;flex-direction:row;align-items:center;gap:5px;min-width:0}
.th-mn{font-size:10px;font-weight:bold;color:var(--bright);flex:1;min-width:0;word-break:break-word;line-height:1.3}
.th-tr{color:var(--orange);margin-left:2px}
.th-model{font-size:8px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100px}
.th-vbars{display:flex;flex-direction:row;align-items:flex-end;gap:3px;flex-shrink:0}
.th-vbg{display:flex;flex-direction:column;align-items:center;gap:1px}
.th-vbt{width:10px;height:52px;background:var(--bg5);border-radius:2px;position:relative;overflow:hidden}
.th-vbf{position:absolute;bottom:0;left:0;right:0;border-radius:2px 2px 0 0}
.th-ba{background:#1e88e5}
.th-bc{background:#90caf9}
.th-vbl{font-size:8px;color:var(--dim);line-height:1}
thead th.dragging{opacity:.35;background:var(--bg5)}
thead th.dov{border-left:2px solid var(--accent)}
thead th.s{cursor:pointer}
thead th.s:hover{background:var(--bg4);color:var(--bright)}
thead th.asc::after{content:" ↑";color:var(--accent)}
thead th.desc::after{content:" ↓";color:var(--accent)}

tbody tr{border-bottom:1px solid var(--bg3)}
tbody tr:hover>td{background:var(--bg2)}
tbody tr.xrow>td{background:var(--bg2)}

td{padding:3px 5px;border-right:1px solid var(--bg3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;vertical-align:top;font-size:11px}
td.c-tog{text-align:center;cursor:pointer;color:var(--dim);padding-top:2px;font-size:13px}
td.c-tog:hover{color:var(--accent)}
td.c-idx{color:var(--dim);text-align:right}
td.c-len{color:var(--dim);text-align:right}
td.c-dis{font-weight:bold}
.dh{color:var(--pink)} .dm{color:var(--orange)} .dl{color:var(--dim)}
.ah{color:var(--green)} .am{color:var(--orange)} .al{color:var(--pink)}
td.c-tgt{color:var(--bright)}
td.c-txt{color:var(--dim);font-style:italic}
.pm{color:#4caf50} .pu{color:var(--dim)} .px{color:#f44336}
.sh{color:#4caf50} .sl{color:#f44336}
.cph{color:#a5d6a7} .cpl{color:#ef9a9a} .cpm{color:var(--dim)}
.bdg{display:inline-block;font-size:8px;padding:0 3px;border-radius:2px;background:var(--bg4);font-weight:bold;margin-right:2px;vertical-align:middle;white-space:nowrap}
.pred-txt{vertical-align:middle}

/* resize handle */
.rh{position:absolute;right:0;top:0;bottom:0;width:5px;cursor:col-resize;background:transparent;z-index:1}
.rh:hover,.rh.dr{background:var(--accent);opacity:.5}

/* expanded detail */
tr.drow>td{padding:0;background:var(--bg2);white-space:normal;border:none}
.dbox{padding:8px 10px 10px 28px;border-left:2px solid var(--accent);margin:2px 4px 4px;overflow-x:auto}
.dtbl-wrap{overflow-x:auto;width:100%}
.dtbl{border-collapse:collapse;font-size:11px;table-layout:fixed;width:100%}
.dtbl th{color:var(--dim);font-size:10px;text-transform:uppercase;padding:2px 5px;text-align:left;background:var(--bg3);white-space:nowrap;position:relative;overflow:hidden}
.dtbl td{padding:2px 5px;border-top:1px solid var(--bg3);vertical-align:top}
.dtbl tr:nth-child(even) td{background:var(--bg3)}
.dtbl td.pred-cell{max-width:400px;white-space:pre-wrap;word-break:break-word;font-size:10px}
.sbar{display:inline-block;height:6px;border-radius:2px;vertical-align:middle;margin-left:3px}
.rsn{color:var(--dim);font-style:italic;font-size:10px;white-space:pre-wrap;word-break:break-word}
.sae-btn{font-size:9px;padding:1px 5px;cursor:pointer;background:var(--bg3);border:1px solid var(--border);border-radius:3px;color:var(--dim)}
.sae-btn:hover{background:var(--border);color:var(--accent)}
.sae-inline{margin-top:4px;background:var(--bg);border:1px solid var(--border);border-radius:3px;padding:5px 7px;font-size:10px;white-space:pre-wrap;max-height:300px;overflow-y:auto;color:var(--dim)}
.sae-modal{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.6);z-index:1000;display:flex;align-items:center;justify-content:center}
.sae-modal-box{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:16px;max-width:700px;width:90%;max-height:80vh;overflow-y:auto;font-size:11px;white-space:pre-wrap;font-family:monospace;position:relative}
.sae-modal-close{position:sticky;top:0;float:right;cursor:pointer;font-size:14px;background:var(--bg2);border:none;color:var(--dim);padding:2px 6px}
#empty{padding:40px;text-align:center;color:var(--dim)}
</style>
</head>
<body>
<div id="hdr">
  <h1>EVAL VIEWER</h1>
  <select id="tsel"><option value="">— task —</option></select>
  <input type="text" id="search" placeholder="filter…" oninput="rerender();_tsave()">

  <div class="row">
    <span class="lbl">sort:</span>
    <button class="btn on" id="sb-agree" onclick="setSort('agree')">agree↓</button>
    <button class="btn"    id="sb-idx"   onclick="setSort('idx')">index</button>
  </div>

  <div class="row" id="cmp-row" style="display:none">
    <span class="lbl">compare:</span>
    <div id="mcbs"></div>
    <button class="btn sm" onclick="mcbAll()">all</button>
    <button class="btn sm" onclick="mcbNone()">none</button>
    <span id="cmp-judge-lbl" style="font-size:10px;color:var(--dim)"></span>
  </div>

  <div class="row">
    <span class="lbl">show cols:</span>
    <div id="tcols"></div>
  </div>

  <span id="stats"></span>
  <div style="font-size:10px;color:var(--dim);margin-top:4px;line-height:1.7">
    <div>
      <span style="color:var(--orange);font-size:12px">†</span> trained on this task &nbsp;|&nbsp;
      <span style="display:inline-block;width:10px;height:8px;background:var(--accent);vertical-align:middle"></span> abs score &nbsp;|&nbsp;
      cell text color = absolute score:&nbsp;
      <span class="pm">&#9632;</span>&nbsp;correct &nbsp;
      <span class="px">&#9632;</span>&nbsp;wrong &nbsp;
      <span class="pu">&#9632;</span>&nbsp;unavailable
    </div>
    <div>
      <b>agr</b> = fraction of methods sharing the majority raw prediction text &nbsp;|&nbsp;
      <span class="bdg sh">abs:1.0</span> = per-item score from eval cache; bright green/red
    </div>
  </div>
</div>
<div id="wrap"><div id="empty">Select a task</div></div>

<script>
let recs=[], methodCols=[], sortKey='idx', sortDir=-1, filterQ='', expanded=new Set(), absType='acc', fieldMapping={};
let dtblColWidths={};
let visTextCols=new Set(['target_response']);
let visMethodCols=new Set();
// Per-method annotation columns shown alongside each method prediction
const ANNOT_COLS=['abs_score','scorer_justification','judge_score'];
const ANNOT_LABELS={abs_score:'abs score',scorer_justification:'scorer justification',judge_score:'judge score'};
let visAnnotCols=new Set(['abs_score','judge_score']);
let colWidths={};
let currentCols=[];
let colOrderOverride=[];
let dragSrcCi=null;
let resizingActive=false;
let methodScores={};
let currentTask='';
let scoreModel='';
let scoreLabel='llm-score';

// Summary table columns (fixed set for the compact table)
const TEXT_COLS=[{key:'question', label:'question'},{key:'supervisor_context', label:'supervisor_context'},{key:'task_prompt', label:'task_prompt'},{key:'target_response', label:'target_response'}];
// Priority text fields shown first in detail view; remaining extra fields shown after
const TEXT_COLS_PRIORITY=['question','supervisor_context','task_prompt','target_response'];
function getTextCols(r){
  const seen=new Set();
  const cols=[];
  // Priority fields first
  TEXT_COLS_PRIORITY.forEach(k=>{if(r[k]!==undefined){cols.push({key:k,label:k});seen.add(k);}});
  // Then any remaining string fields (skip internal _ fields and method predictions)
  const skip=new Set([...seen,'example_id','_abs_scores','_scorer_responses','_prompts','_masked_supervisor_contexts','_activation_summaries','_best_method','_compare_justification','_pred_agreement','llm_comparative_score','prompt',...methodCols]);
  Object.keys(r).forEach(k=>{if(!skip.has(k)&&!k.startsWith('_')&&typeof r[k]==='string'&&r[k].length>0&&r[k].length<10000)cols.push({key:k,label:k});});
  return cols;
}

// ── Persistence ───────────────────────────────────────────────────────
function _gsave(){
  localStorage.setItem('ev_g',JSON.stringify({task:currentTask,sk:sortKey,sd:sortDir}));
}
function _tsave(){
  if(!currentTask)return;
  localStorage.setItem('ev_t_'+currentTask,JSON.stringify({
    vtc:[...visTextCols],vmc:[...visMethodCols],vac:[...visAnnotCols],cw:colWidths,co:colOrderOverride,
    mc:methodCols,
    ex:[...expanded],sel:Array.from(document.querySelectorAll('#mcbs input:checked')).map(x=>x.value),
    q:document.getElementById('search').value,sc:document.getElementById('wrap').scrollTop,
  }));
}
function _trestore(){
  try{const r=localStorage.getItem('ev_t_'+currentTask);return r?JSON.parse(r):null;}catch{return null;}
}
function _applyTaskState(s){
  if(!s)return;
  if(s.vtc)visTextCols=new Set(s.vtc);
  if(s.vac)visAnnotCols=new Set(s.vac);
  if(s.vmc&&s.mc){
    const prevMethods=new Set(s.mc);
    visMethodCols=new Set(s.vmc.filter(m=>methodCols.includes(m)));
    methodCols.forEach(m=>{if(!prevMethods.has(m))visMethodCols.add(m);});
  }
  if(s.cw)colWidths=s.cw;
  if(s.co)colOrderOverride=s.co;
  if(s.ex)expanded=new Set(s.ex);
  if(s.q){const el=document.getElementById('search');if(el)el.value=s.q;}
  if(s.sel){
    document.querySelectorAll('#mcbs input').forEach(cb=>{
      const on=s.sel.includes(cb.value);cb.checked=on;
      document.getElementById('ml-'+cb.value)?.classList.toggle('on',on);
    });
  }
  TEXT_COLS.forEach(({key})=>{
    const lbl=document.getElementById('tl-'+key);
    if(lbl){const on=visTextCols.has(key);lbl.classList.toggle('on',on);lbl.querySelector('input').checked=on;}
  });
  methodCols.forEach(m=>{
    const lbl=document.getElementById('tl-m-'+m);
    if(lbl){const on=visMethodCols.has(m);lbl.classList.toggle('on',on);lbl.querySelector('input').checked=on;}
  });
  ANNOT_COLS.forEach(a=>{
    const lbl=document.getElementById('tl-a-'+a);
    if(lbl){const on=visAnnotCols.has(a);lbl.classList.toggle('on',on);lbl.querySelector('input').checked=on;}
  });
}

// ── Init ──────────────────────────────────────────────────────────────
(async()=>{
  const g=(()=>{try{const r=localStorage.getItem('ev_g');return r?JSON.parse(r):null;}catch{return null;}})();
  if(g){if(g.sk)sortKey=g.sk;if(g.sd!=null)sortDir=g.sd;}
  const ts=await(await fetch('/api/tasks')).json();
  const sel=document.getElementById('tsel');
  ts.forEach(t=>{const o=document.createElement('option');o.value=o.textContent=t;sel.appendChild(o)});
  sel.addEventListener('change',()=>loadTask(sel.value));
  renderTextColToggles();
  document.getElementById('wrap').addEventListener('scroll',_tsave,{passive:true});
  if(g?.task&&ts.includes(g.task)){sel.value=g.task;await loadTask(g.task);}
})();

// ── Load ──────────────────────────────────────────────────────────────
async function loadTask(task){
  if(!task)return;
  currentTask=task;
  expanded.clear();
  colWidths={};
  colOrderOverride=[];
  methodScores={};
  document.getElementById('wrap').innerHTML='<div id="empty">Loading…</div>';
  const [recResp, methodScoresResp]=await Promise.all([
    fetch('/api/records/'+task).then(r=>r.json()),
    fetch('/api/method_scores/'+task).then(r=>r.json()).catch(()=>({}))
  ]);
  recs=recResp.records; absType=recResp.abs_type||'acc'; scoreModel=recResp.score_model||''; scoreLabel=recResp.score_label||'llm-score'; fieldMapping=recResp.field_mapping||{}; methodScores=methodScoresResp;
  const judgeModel=recResp.judge_model||scoreModel||'';
  const _jmShort=judgeModel?judgeModel.split('/').pop():'?';
  const cmpJudgeLbl=document.getElementById('cmp-judge-lbl');
  if(cmpJudgeLbl)cmpJudgeLbl.textContent=judgeModel?`judge: ${_jmShort}`:'';

  recs.forEach(r=>{r._len=(r.target_response||'').length;});
  methodCols=inferMethods(recs, methodScores);
  visMethodCols=new Set(methodCols);
  renderMcbs();
  renderMethodColToggles();
  const saved=_trestore();
  _applyTaskState(saved);
  rerender();
  if(saved?.sc)document.getElementById('wrap').scrollTop=saved.sc;
  _gsave();
}

function inferMethods(rs, scoreMap){
  const ord=['weak-bb-monitor','strong-bb-monitor','original_ao','our_ao','linear_probes','sae-llm-monitor','patchscopes'];
  const have=new Set();
  rs.forEach(r=>ord.forEach(m=>{if(m in r)have.add(m)}));
  Object.keys(scoreMap||{}).forEach(m=>{if(ord.includes(m))have.add(m)});
  return ord.filter(m=>have.has(m));
}

// ── Column visibility toggles ────────────────────────────────────────
function _colLabel(key,label){
  const mapped=fieldMapping[key];
  return mapped&&mapped!==key?`${label} (${mapped})`:label;
}
function renderTextColToggles(){
  const c=document.getElementById('tcols');
  const textHtml=TEXT_COLS.map(({key,label})=>{
    const on=visTextCols.has(key)?'on':'';
    return `<label class="tcol-lbl ${on}" id="tl-${key}"><input type="checkbox" value="${key}" ${on?'checked':''} onchange="onTcol(this)"> ${_colLabel(key,label)}</label>`;
  }).join('');
  c.innerHTML=textHtml+'<span id="meth-col-toggles"></span>';
}

function renderMethodColToggles(){
  const span=document.getElementById('meth-col-toggles');
  if(!span)return;
  const methHtml=(methodCols.length?'<span style="color:var(--dim);font-size:10px;margin:0 4px">|</span>':'')+
    methodCols.map(m=>{
      const on=visMethodCols.has(m)?'on':'';
      return `<label class="tcol-lbl ${on}" id="tl-m-${m}"><input type="checkbox" value="${m}" ${on?'checked':''} onchange="onMcol(this)"> ${sn(m)}</label>`;
    }).join('');
  const annotHtml='<span style="color:var(--dim);font-size:10px;margin:0 4px">|</span>'+
    ANNOT_COLS.map(a=>{
      const on=visAnnotCols.has(a)?'on':'';
      return `<label class="tcol-lbl ${on}" id="tl-a-${a}"><input type="checkbox" value="${a}" ${on?'checked':''} onchange="onAnnotCol(this)"> ${ANNOT_LABELS[a]}</label>`;
    }).join('');
  span.innerHTML=methHtml+annotHtml;
}

function onTcol(cb){
  if(cb.checked) visTextCols.add(cb.value); else visTextCols.delete(cb.value);
  document.getElementById('tl-'+cb.value)?.classList.toggle('on',cb.checked);
  rerender();_tsave();
}

function onMcol(cb){
  if(cb.checked) visMethodCols.add(cb.value); else visMethodCols.delete(cb.value);
  document.getElementById('tl-m-'+cb.value)?.classList.toggle('on',cb.checked);
  rerender();_tsave();
}

function onAnnotCol(cb){
  if(cb.checked) visAnnotCols.add(cb.value); else visAnnotCols.delete(cb.value);
  document.getElementById('tl-a-'+cb.value)?.classList.toggle('on',cb.checked);
  rerender();_tsave();
}

// ── Method checkboxes ────────────────────────────────────────────────
const DEFAULT_CMP_METHODS=new Set(['our_ao','weak-bb-monitor']);
function _mcbLabel(m){
  const base=sn(m);
  if(m==='weak-bb-monitor'||m==='strong-bb-monitor'){
    const model=methodScores[m]?.model||'';
    const short=model.split('/').pop();
    if(short)return`${base} <span style="color:var(--dim);font-size:9px">(${short})</span>`;
  }
  return base;
}
function renderMcbs(){
  const c=document.getElementById('mcbs');
  c.innerHTML=methodCols.map(m=>{const on=DEFAULT_CMP_METHODS.has(m)?'on':'';const chk=DEFAULT_CMP_METHODS.has(m)?'checked':'';return `<label class="mcb-lbl ${on}" id="ml-${m}"><input type="checkbox" value="${m}" ${chk} onchange="onMcb(this)"> ${_mcbLabel(m)}</label>`;}).join('');
  document.getElementById('cmp-row').style.display='flex';
}

function onMcb(cb){
  document.getElementById('ml-'+cb.value)?.classList.toggle('on',cb.checked);
  rerender();_tsave();
}
function mcbAll(){document.querySelectorAll('#mcbs input').forEach(cb=>{cb.checked=true;document.getElementById('ml-'+cb.value)?.classList.add('on')});rerender();_tsave();}
function mcbNone(){document.querySelectorAll('#mcbs input').forEach(cb=>{cb.checked=false;document.getElementById('ml-'+cb.value)?.classList.remove('on')});rerender();_tsave();}
function selMethods(){const c=Array.from(document.querySelectorAll('#mcbs input:checked')).map(x=>x.value);return c.length?c:methodCols}

// ── Render ────────────────────────────────────────────────────────────
function rerender(){
  const q=document.getElementById('search').value.toLowerCase();
  filterQ=q;
  const filtered=q?recs.filter(r=>JSON.stringify(r).toLowerCase().includes(q)):recs;
  const sorted=[...filtered].sort((a,b)=>{
    let av,bv;
    if(sortKey==='agree'){av=a._pred_agreement||0;bv=b._pred_agreement||0}
    else if(sortKey==='idx'){av=recs.indexOf(a);bv=recs.indexOf(b)}
    else if(sortKey==='len'){av=a._len||0;bv=b._len||0}
    else if(sortKey.startsWith('m:')){const m=sortKey.slice(2);av=String(a[m]??'');bv=String(b[m]??'')}
    else if(sortKey.startsWith('t:')){const k=sortKey.slice(2);av=String(a[k]??'');bv=String(b[k]??'')}
    else return 0;
    if(av<bv)return-sortDir;if(av>bv)return sortDir;return 0;
  });
  document.getElementById('stats').textContent=`${sorted.length}/${recs.length}`;

  const cols=buildCols();
  currentCols=cols;
  const tw=cols.reduce((s,c)=>s+(colWidths[c.id]??c.defaultW),0);
  let h=[`<table id="main-tbl" style="table-layout:fixed;width:${tw}px"><colgroup>`];
  cols.forEach(c=>h.push(`<col class="${c.cc}" style="width:${colWidths[c.id]??c.defaultW}px">`));
  h.push('</colgroup><thead><tr>');
  cols.forEach((c,i)=>{
    const sc=c.sk?` s`:'', so=c.sk===sortKey?(sortDir<0?' desc':' asc'):'', oc=c.sk?` onclick="ts('${c.sk}')"`:''
    const lbl=c.method?thMethodLbl(c.method):c.lbl;
    h.push(`<th class="${c.cc}${sc}${so}"${oc} data-ci="${i}" draggable="true" ondragstart="thDs(event,${i})" ondragend="thDe(event)" ondragenter="thDen(event)" ondragleave="thDlv(event)" ondragover="thDov(event)" ondrop="thDrop(event,${i})">${lbl}<span class="rh" onpointerdown="sr(event,${i})" onclick="event.stopPropagation()"></span></th>`);
  });
  h.push('</tr></thead><tbody>');
  sorted.forEach((r,vi)=>{
    const oi=recs.indexOf(r);
    h.push(rowHtml(r,oi,vi,cols));
    if(expanded.has(oi))h.push(detailHtml(r,cols.length,oi));
  });
  h.push('</tbody></table>');
  document.getElementById('wrap').innerHTML=h.join('');
  // Auto-trigger comparative scoring for newly expanded items
  document.querySelectorAll('[data-auto="1"]').forEach(el=>{
    el.dataset.auto='0';
    runJudge(el.dataset.task, parseInt(el.dataset.idx), el.id);
  });
}

function buildCols(){
  const cols=[
    {id:'idx', cc:'c-idx', lbl:'eval_id', sk:'idx', defaultW:140},
    {id:'tog', cc:'c-tog', lbl:'',        sk:null,  defaultW:22},
    {id:'ag',  cc:'c-ag',  lbl:'agr',     sk:'agree', defaultW:50},
    {id:'len', cc:'c-len', lbl:'chars',   sk:'len', defaultW:58},
  ];
  TEXT_COLS.forEach(({key,label})=>{
    if(!visTextCols.has(key))return;
    cols.push({id:'t:'+key, cc:'c-txt', lbl:_colLabel(key,label), sk:'t:'+key, defaultW:180});
  });
  methodCols.filter(m=>visMethodCols.has(m)).forEach(m=>cols.push({id:'m:'+m, cc:'c-meth', lbl:sn(m), sk:'m:'+m, defaultW:140, method:m}));
  if(colOrderOverride.length){
    const pos=new Map(colOrderOverride.map((id,i)=>[id,i]));
    cols.sort((a,b)=>(pos.get(a.id)??999)-(pos.get(b.id)??999));
  }
  return cols;
}

function rowHtml(r,oi,vi,cols){
  const isX=expanded.has(oi);
  const ag=r._pred_agreement||0, acl=ag>.85?'ah':ag>.6?'am':'al';
  let h=`<tr class="${isX?'xrow':''}" id="row-${oi}">`;
  h+=`<td class="c-idx" title="${esc(String(r.example_id??''))}">${esc(String(r.example_id??''))}</td>`;
  h+=`<td class="c-tog" onclick="tog(${oi})">${isX?'▼':'▶'}</td>`;
  h+=`<td class="c-ag  ${acl}" title="${(ag*100).toFixed(0)}%">${(ag*100).toFixed(0)}%</td>`;
  h+=`<td class="c-len" title="target chars">${r._len??0}</td>`;
  TEXT_COLS.forEach(({key})=>{
    if(!visTextCols.has(key))return;
    const v=String(r[key]||'');
    h+=`<td class="c-txt" title="${esc(v)}">${esc(v)}</td>`;
  });
  methodCols.filter(m=>visMethodCols.has(m)).forEach(m=>{
    const v=r[m]!==undefined?String(r[m]):'—';
    let badges='';
    if(visAnnotCols.has('abs_score')){
      const av=r._abs_scores?.[m];
      if(av!==undefined){const acl=av>=0.5?'sh':'sl'; badges+=`<span class="bdg ${acl}">abs:${av.toFixed(1)}</span>`;}
    }
    if(visAnnotCols.has('judge_score')){
      const jv=r.llm_comparative_score?.[m];
      if(jv!==undefined){const jcl=jv>=0.5?'sh':'sl'; badges+=`<span class="bdg ${jcl}">judge:${jv.toFixed(1)}</span>`;}
    }
    const tip=esc(v);
    h+=`<td class="c-meth" title="${tip}">${badges}<span class="pred-txt">${esc(v)}</span></td>`;
  });
  h+='</tr>';
  return h;
}

function detailHtml(r,colspan,oi){
  let h=`<tr class="drow"><td colspan="${colspan}"><div class="dbox">`;

  // Text fields grid (dynamic per-record)
  const txts=getTextCols(r);
  if(txts.length){
    h+=`<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;margin-bottom:8px">`;
    txts.forEach(({key,label})=>{
      const v=esc(String(r[key]));
      h+=`<div><div style="font-size:9px;text-transform:uppercase;color:var(--dim);margin-bottom:2px">${_colLabel(key,label)}</div>
          <div style="background:var(--bg);border:1px solid var(--border);border-radius:2px;padding:5px 7px;font-size:11px;line-height:1.5;max-height:180px;overflow-y:auto;white-space:pre-wrap;word-break:break-word">${v}</div></div>`;
    });
    h+=`</div>`;
  }

  // Methods table — dynamic columns based on annotation toggles
  const hasPreds=methodCols.some(m=>r[m]!==undefined);
  if(hasPreds){
    const dcols=[{id:'method',lbl:'method',dw:110}];
    dcols.push({id:'abs_score',lbl:'abs',dw:50},{id:'bar',lbl:'',dw:70});
    if(visAnnotCols.has('judge_score'))dcols.push({id:'judge_score',lbl:'judge',dw:50});
    if(visAnnotCols.has('scorer_justification'))dcols.push({id:'scorer_justification',lbl:'scorer justification',dw:280});
    dcols.push({id:'pred',lbl:'prediction',dw:420});
    const cg=`<colgroup>${dcols.map(c=>`<col data-dc="${c.id}" style="width:${dtblColWidths[c.id]??c.dw}px">`).join('')}</colgroup>`;
    const hdr=dcols.map(c=>`<th>${c.lbl}<div class="rh" onpointerdown="dsr(event,'${c.id}')"></div></th>`).join('');
    h+=`<div class="dtbl-wrap"><table class="dtbl">${cg}<tr>${hdr}</tr>`;
    methodCols.forEach(m=>{
      if(r[m]===undefined)return;
      const av=r._abs_scores?.[m];
      const scoreCell=av!==undefined?(()=>{
        const cl=av>=.7?'sh':av>=.4?'sm_':'sl';
        const bc=av>=.7?'#7ec480':av>=.4?'#cc8a40':'#c97b7b';
        const bw=Math.round(av*80);
        return `<td><span class="sv ${cl}">${av.toFixed(3)}</span></td><td><span class="sbar" style="width:${bw}px;background:${bc}"></span></td>`;
      })():`<td>—</td><td></td>`;
      let extraCells='';
      if(visAnnotCols.has('judge_score')){
        const jv=r.llm_comparative_score?.[m];
        if(jv!==undefined){const jcl=jv>=.5?'sh':'sl'; extraCells+=`<td><span class="sv ${jcl}">${jv.toFixed(1)}</span></td>`;}
        else extraCells+=`<td>—</td>`;
      }
      if(visAnnotCols.has('scorer_justification')){
        const sr=r._scorer_responses?.[m];
        extraCells+=`<td class="pred-cell" style="font-size:10px;color:var(--dim);max-width:280px;overflow:hidden;text-overflow:ellipsis" title="${sr?esc(sr):''}">${sr?esc(sr.slice(0,200)):'—'}</td>`;
      }
      const pred=esc(String(r[m]));
      const saeCid=`sae-${esc(String(r.example_id??''))}`;
      const isSae=m==='sae-llm-monitor';
      const saeBtn=isSae?` <button class="sae-btn" id="${saeCid}-btn" onclick="toggleSaeInline('${currentTask}','${esc(String(r.example_id??''))}','${saeCid}')">▶ features</button>`:'';
      const itemPrompt=r._prompts?.[m];
      const promptRow=itemPrompt?`<div style="margin-top:3px;padding:3px 5px;background:var(--bg);border:1px solid var(--border);border-radius:2px;font-size:9px;color:var(--dim);white-space:pre-wrap;max-height:120px;overflow-y:auto"><b>prompt:</b> ${esc(itemPrompt)}</div>`:'';
      const maskedCtx=r._masked_supervisor_contexts?.[m];
      const maskedRow=maskedCtx?`<div style="margin-top:3px;padding:3px 5px;background:var(--bg);border:1px solid var(--border);border-radius:2px;font-size:9px;color:var(--dim);white-space:pre-wrap;max-height:120px;overflow-y:auto"><b>masked context:</b> ${esc(maskedCtx)}</div>`:'';
      const actSummary=r._activation_summaries?.[m];
      const actRow=actSummary?`<div style="margin-top:3px;padding:3px 5px;background:var(--bg);border:1px solid var(--border);border-radius:2px;font-size:9px;color:var(--dim);white-space:pre-wrap;max-height:80px;overflow-y:auto"><b>activation summary:</b> ${esc(actSummary)}</div>`:'';
      h+=`<tr><td style="white-space:nowrap">${m}</td>${scoreCell}${extraCells}<td class="pred-cell">${promptRow}${maskedRow}${actRow}${pred}${isSae?`<div id="${saeCid}"></div>`:''}${saeBtn}</td></tr>`;
    });
    h+=`</table></div>`;
  }

  // Comparative scoring — auto-triggered on expand
  const cmpId=`cmp-${oi}`;
  h+=`<div id="${cmpId}" data-auto="1" data-task="${currentTask}" data-idx="${oi}" style="margin-top:8px;padding:6px 8px;background:var(--bg);border:1px solid var(--border);border-radius:3px;font-size:11px">`;
  h+=`<div style="font-size:9px;text-transform:uppercase;color:var(--dim);margin-bottom:3px">judge scoring</div>`;
  h+=`<span style="color:var(--dim)">loading…</span>`;
  h+=`</div>`;

  h+=`</div></td></tr>`;
  return h;
}

// ── Toggle / sort ─────────────────────────────────────────────────────
function tog(oi){expanded.has(oi)?expanded.delete(oi):expanded.add(oi);rerender();_tsave();}
function setSort(k){sortDir=(sortKey===k)?-sortDir:-1;sortKey=k;['agree','idx'].forEach(x=>document.getElementById('sb-'+x)?.classList.toggle('on',x===k));rerender();_gsave();}

async function runJudge(task, itemIdx, containerId){
  const box=document.getElementById(containerId);
  if(!box)return;
  box.innerHTML=`<div style="font-size:9px;text-transform:uppercase;color:var(--dim);margin-bottom:3px">judge scoring</div><span style="color:var(--dim)">calling judge…</span>`;
  try{
    const r=await fetch(`/api/compare/${encodeURIComponent(task)}/${itemIdx}`);
    const d=await r.json();
    if(d.error){box.innerHTML+=`<div style="color:#c66">${esc(d.error)}</div>`;return;}
    // Store judge results back into the record so table cells update on rerender
    if(recs[itemIdx]){
      recs[itemIdx].llm_comparative_score=d.method_scores||{};
      recs[itemIdx]._compare_justification=d.justification||'';
      recs[itemIdx]._best_method=d.best_method||'';
    }
    const scTxt=Object.entries(d.method_scores||{}).map(([m,s])=>`${m}=${s.toFixed(1)}`).join(', ');
    let h=`<div style="font-size:9px;text-transform:uppercase;color:var(--dim);margin-bottom:3px">judge scoring</div>`;
    if(d.best_method) h+=`<div><b>best:</b> ${esc(d.best_method)}${scTxt?' &nbsp;|&nbsp; '+esc(scTxt):''}</div>`;
    if(d.justification) h+=`<div style="margin-top:3px;color:var(--fg);line-height:1.4">${esc(d.justification)}</div>`;
    box.innerHTML=h;
  }catch(e){box.innerHTML+=`<div style="color:#c66">${e}</div>`;}
}

function toggleSaeInline(task, exampleId, containerId){
  const box=document.getElementById(containerId);
  const btn=document.getElementById(containerId+'-btn');
  if(!box||!btn)return;
  if(box.dataset.loaded){box.style.display=box.style.display==='none'?'block':'none';btn.textContent=box.style.display==='none'?'▶ features':'▼ features';return;}
  btn.textContent='…';
  fetch(`/api/sae_features/${encodeURIComponent(task)}/${encodeURIComponent(exampleId)}`)
    .then(r=>r.json())
    .then(d=>{
      const text=d.error?d.error:(d.feature_desc||'(no feature_desc)');
      const prompt=d.full_prompt?`\n\n--- FULL PROMPT ---\n${d.full_prompt}`:'';
      box.className='sae-inline';
      box.textContent=text+prompt;
      box.dataset.loaded='1';
      btn.textContent='▼ features';
    })
    .catch(e=>{btn.textContent='▶ features';box.textContent='Error: '+e;box.className='sae-inline'});
}
function ts(k){setSort(k)}

// ── Column resize ─────────────────────────────────────────────────────
function sr(e,ci){
  e.stopPropagation();e.preventDefault();
  const th=e.target.closest('th'), handle=e.target;
  handle.setPointerCapture(e.pointerId);
  const startX=e.clientX, startW=th.offsetWidth;
  handle.classList.add('dr');
  const cid=currentCols[ci]?.id;
  const tbl=document.getElementById('main-tbl');
  const col=tbl?.querySelectorAll(':scope>colgroup>col')[ci];
  const mv=ev=>{
    const nw=Math.max(20,startW+(ev.clientX-startX));
    if(cid)colWidths[cid]=nw;
    if(col)col.style.width=nw+'px';
    if(tbl)tbl.style.width=currentCols.reduce((s,c)=>s+(colWidths[c.id]??c.defaultW),0)+'px';
  };
  const up=()=>{resizingActive=false;handle.classList.remove('dr');handle.removeEventListener('pointermove',mv);handle.removeEventListener('pointerup',up);_tsave();};
  handle.addEventListener('pointermove',mv);
  handle.addEventListener('pointerup',up);
}

function dsr(e,colId){
  e.stopPropagation();e.preventDefault();
  const th=e.target.closest('th'),handle=e.target;
  handle.setPointerCapture(e.pointerId);
  const startX=e.clientX,startW=th.offsetWidth;
  handle.classList.add('dr');
  const mv=ev=>{
    const nw=Math.max(30,startW+(ev.clientX-startX));
    dtblColWidths[colId]=nw;
    document.querySelectorAll(`.dtbl col[data-dc="${colId}"]`).forEach(c=>c.style.width=nw+'px');
  };
  const up=()=>{handle.classList.remove('dr');handle.removeEventListener('pointermove',mv);handle.removeEventListener('pointerup',up);};
  handle.addEventListener('pointermove',mv);
  handle.addEventListener('pointerup',up);
}

// ── Column drag-to-reorder ─────────────────────────────────────────────
function thDs(e,ci){
  if(resizingActive){e.preventDefault();return}
  dragSrcCi=ci;
  e.dataTransfer.effectAllowed='move';
  e.dataTransfer.setData('text/plain',String(ci));
  setTimeout(()=>e.target.closest('th')?.classList.add('dragging'),0);
}
function thDe(e){e.target.closest('th')?.classList.remove('dragging');dragSrcCi=null}
function thDen(e){e.preventDefault();e.target.closest('th')?.classList.add('dov')}
function thDlv(e){e.target.closest('th')?.classList.remove('dov')}
function thDov(e){e.preventDefault();e.dataTransfer.dropEffect='move'}
function thDrop(e,ci){
  e.preventDefault();
  e.target.closest('th')?.classList.remove('dov');
  if(dragSrcCi===null||dragSrcCi===ci){dragSrcCi=null;return}
  const ids=currentCols.map(c=>c.id);
  const [rem]=ids.splice(dragSrcCi,1);
  ids.splice(ci,0,rem);
  colOrderOverride=ids;
  dragSrcCi=null;
  rerender();_tsave();
}

// ── Utils ─────────────────────────────────────────────────────────────
function sn(m){return m.replace(/_/g,' ')}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

function thMethodLbl(m){
  const ms=methodScores[m];
  const tr=ms?.trained?`<span class="th-tr" title="trained on this task">†</span>`:'';
  const skipped=ms?.skipped;
  let vbars='';
  if(ms&&!skipped){
    const abs=ms.primary_score??0;
    const vbar=(lbl,val,cls)=>`<div class="th-vbg"><div class="th-vbt"><div class="th-vbf ${cls}" style="height:${(val*100).toFixed(0)}%" title="${lbl}: ${val.toFixed(2)}"></div></div><span class="th-vbl">${lbl}</span></div>`;
    vbars=`<div class="th-vbars">${vbar('abs('+absType+')',abs,'th-ba')}</div>`;
  }
  const modelLbl=ms?.model?`<div class="th-model" title="${esc(ms.model)}">${esc(ms.model.split('/').pop())}</div>`:'';
  const skippedLbl=skipped?`<div class="th-model" title="${esc(ms?.skip_reason||'skipped')}">skipped</div>`:'';
  return `<div class="th-mc"><div class="th-mn">${sn(m)}${tr}</div>${modelLbl}${skippedLbl}${vbars}</div>`;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Eval Viewer — web dashboard for comprehensive eval results")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to eval_cache.db")
    parser.add_argument("--run-id", default=None, help="Specific run_id (default: most recent)")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    _cache = EvalCache(db_path)

    # Clear stale comparative scores — now computed on-the-fly
    _cache._conn.execute("DELETE FROM comparative_scores")
    _cache._conn.commit()

    if args.run_id:
        _run_id = args.run_id
    else:
        # Pick the most recent run
        rows = _cache._conn.execute("SELECT run_id, created_at FROM eval_runs ORDER BY created_at DESC LIMIT 1").fetchall()
        if not rows:
            print("No runs found in DB")
            sys.exit(1)
        _run_id = rows[0][0]
        print(f"Using run_id: {_run_id} (created {rows[0][1]})")

    print(f"DB: {db_path}")
    print(f"Score model (from config): {_SCORE_MODEL}")
    print(f"Serving on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
