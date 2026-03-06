#!/usr/bin/env python3
"""Web viewer for per_example_records.json from comprehensive eval."""

import json
import statistics
import sys
from pathlib import Path

import yaml
from flask import Flask, jsonify

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tasks import TASKS, ScoringMode

_ABS_TYPE = {
    ScoringMode.BINARY: "acc",
    ScoringMode.TOKEN_F1: "f1",
    ScoringMode.LLM_JUDGE: "llm",
    ScoringMode.STEP_ACCURACY: "acc",
    ScoringMode.TOKEN_MATCH: "acc",
}

def _task_abs_type(task: str) -> str:
    td = TASKS.get(task)
    return _ABS_TYPE.get(td.scoring, "acc") if td else "acc"

app = Flask(__name__)
LOGS_DIR = Path(__file__).resolve().parent.parent / "data" / "comprehensive_eval" / "logs"
METHOD_ORDER = ["llm_monitor_pro", "original_ao", "our_ao", "celeste_ao", "linear_probes", "sae_probe"]

_train_cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs/train.yaml").read_text())
OUR_AO_TRAINING_TASKS = {k for k, v in _train_cfg["tasks"].items() if isinstance(v, dict) and v.get("n", 0) != 0}
_og_ao_cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs/og_ao.yaml").read_text())
OG_AO_TRAINING_TASKS = set(_og_ao_cfg.get("trained_tasks", []))
METHOD_TO_SCORE_FILE = {
    "llm_monitor_flash": "llm_monitor_flash", "llm_monitor_pro": "llm_monitor_pro",
    "original_ao": "original_ao_kall", "our_ao": "our_ao_kall", "celeste_ao": "celeste_ao_kall",
    "linear_probes": "linear_probes", "sae_probe": "sae_probe",
}


@app.route("/")
def index():
    return HTML


@app.route("/api/tasks")
def list_tasks():
    tasks = [d.name for d in sorted(LOGS_DIR.iterdir()) if d.is_dir() and (d / "per_example_records.json").exists()]
    return jsonify(tasks)


@app.route("/api/records/<task>")
def get_records(task):
    from collections import Counter
    task_dir = LOGS_DIR / task
    records = json.loads((task_dir / "per_example_records.json").read_text())
    # Load per-example abs scores from task result files
    abs_score_data: dict[str, list] = {}
    for method, fname in METHOD_TO_SCORE_FILE.items():
        p = task_dir / f"{fname}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if "per_example_scores" in d:
                abs_score_data[method] = d["per_example_scores"]
    for i, r in enumerate(records):
        raw = r.get("llm_comparative_score", {})
        normed = {}
        for k, v in raw.items():
            normed[k] = v if isinstance(v, dict) else {"score": float(v), "reason": ""}
        r["llm_comparative_score"] = normed
        vals = [v["score"] for v in normed.values()]
        r["_disagreement"] = round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0
        methods = [k for k in METHOD_ORDER if k in r]
        preds = [str(r[m]).strip().lower()[:40] for m in methods]
        if preds:
            majority = Counter(preds).most_common(1)[0][1]
            r["_pred_agreement"] = round(majority / len(preds), 2)
        else:
            r["_pred_agreement"] = 1.0
        r["_abs_scores"] = {m: scores[i] for m, scores in abs_score_data.items() if i < len(scores)}
    return jsonify({"records": records, "abs_type": _task_abs_type(task)})


@app.route("/api/method_scores/<task>")
def method_scores_api(task):
    log_dir = LOGS_DIR / task
    records = json.loads((log_dir / "per_example_records.json").read_text())
    comp_sums, comp_counts = {}, {}
    for r in records:
        for m, v in r.get("llm_comparative_score", {}).items():
            s = v["score"] if isinstance(v, dict) else float(v)
            comp_sums[m] = comp_sums.get(m, 0) + s
            comp_counts[m] = comp_counts.get(m, 0) + 1
    result = {}
    for method, fname in METHOD_TO_SCORE_FILE.items():
        path = log_dir / f"{fname}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        is_cls_task = task.startswith("cls_")
        trained = (method == "our_ao" and task in OUR_AO_TRAINING_TASKS) or \
                  (method == "original_ao" and (task in OG_AO_TRAINING_TASKS or is_cls_task))
        comp = comp_sums[method] / comp_counts[method] if method in comp_counts else None
        result[method] = {"primary_score": data["primary_score"], "comparative_score": comp, "trained": trained, "model": data.get("model")}
    return jsonify(result)


@app.route("/api/sae_features/<task>/<example_id>")
def sae_features_api(task, example_id):
    path = LOGS_DIR / task / "sae_probe_features.jsonl"
    if not path.exists():
        return jsonify({"error": "no sae_probe_features.jsonl for this task"}), 404
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if str(rec.get("example_id")) == str(example_id):
            return jsonify(rec)
    return jsonify({"error": f"example_id {example_id!r} not found"}), 404


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
.rsn{color:var(--dim);font-style:italic;font-size:10px}
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
    <button class="btn on" id="sb-dis"   onclick="setSort('dis')">disagree↓</button>
    <button class="btn"    id="sb-agree" onclick="setSort('agree')">agree</button>
    <button class="btn"    id="sb-idx"   onclick="setSort('idx')">index</button>
  </div>

  <div class="row" id="cmp-row" style="display:none">
    <span class="lbl">compare:</span>
    <div id="mcbs"></div>
    <button class="btn sm" onclick="mcbAll()">all</button>
    <button class="btn sm" onclick="mcbNone()">none</button>
  </div>

  <div class="row">
    <span class="lbl">show cols:</span>
    <div id="tcols"></div>
  </div>

  <span id="stats"></span>
  <div style="font-size:10px;color:var(--dim);margin-top:4px;line-height:1.7">
    <div>
      <span style="color:var(--orange);font-size:12px">†</span> trained on this task &nbsp;|&nbsp;
      <span style="display:inline-block;width:10px;height:8px;background:var(--accent);vertical-align:middle"></span> abs score &nbsp;
      <span style="display:inline-block;width:10px;height:8px;background:var(--green);vertical-align:middle"></span> cmp score &nbsp;|&nbsp;
      cell text color = absolute score:&nbsp;
      <span class="pm">&#9632;</span>&nbsp;correct &nbsp;
      <span class="px">&#9632;</span>&nbsp;wrong &nbsp;
      <span class="pu">&#9632;</span>&nbsp;unavailable
    </div>
    <div>
      <b>dis</b> = stdev of comparative scores across selected methods (high → methods disagree) &nbsp;|&nbsp;
      <b>agr</b> = fraction of methods sharing the majority raw prediction text &nbsp;|&nbsp;
      <span class="bdg sh">abs:1</span> = absolute score (scorer type shown in column header bar); bright green/red &nbsp;|&nbsp;
      <span class="bdg cph">cmp:.9</span>/<span class="bdg cpl">cmp:.1</span> = comparative LLM score; pale green/red
    </div>
  </div>
</div>
<div id="wrap"><div id="empty">Select a task</div></div>

<script>
let recs=[], methodCols=[], sortKey='dis', sortDir=-1, filterQ='', expanded=new Set(), absType='acc';
let dtblColWidths={};
let visTextCols=new Set(['target_response']);
let visMethodCols=new Set();   // populated on task load; excludes pro by default
let colWidths={};              // col.id -> px, persists across rerenders
let currentCols=[];            // cols from last buildCols() call
let colOrderOverride=[];       // ordered col ids; [] = natural order
let dragSrcCi=null;
let resizingActive=false;
let methodScores={};
let currentTask='';

const TEXT_COLS=[
  {key:'question',          label:'question'},
  {key:'cot_field',         label:'cot_field'},
  {key:'cot_suffix',        label:'cot_suffix'},
  {key:'masked_cot_field',  label:'masked_cot_field'},
  {key:'oracle_prefix',     label:'oracle prefix'},
  {key:'prompt',            label:'prompt'},
  {key:'target_response',   label:'target_response'},
];

// ── Persistence ───────────────────────────────────────────────────────
function _gsave(){
  localStorage.setItem('ev_g',JSON.stringify({task:currentTask,sk:sortKey,sd:sortDir}));
}
function _tsave(){
  if(!currentTask)return;
  localStorage.setItem('ev_t_'+currentTask,JSON.stringify({
    vtc:[...visTextCols],vmc:[...visMethodCols],cw:colWidths,co:colOrderOverride,
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
  if(s.vmc)visMethodCols=new Set(s.vmc.filter(m=>methodCols.includes(m)));
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
  recs=recResp.records; absType=recResp.abs_type||'acc'; methodScores=methodScoresResp;
  methodCols=inferMethods(recs);
  visMethodCols=new Set(methodCols.filter(m=>m!=='llm_monitor_pro'));
  renderMcbs();
  renderMethodColToggles();
  const saved=_trestore();
  _applyTaskState(saved);
  rerender();
  if(saved?.sc)document.getElementById('wrap').scrollTop=saved.sc;
  _gsave();
}

function inferMethods(rs){
  const ord=['llm_monitor_flash','llm_monitor_pro','original_ao','our_ao','celeste_ao','linear_probes','sae_probe'];
  const have=new Set();
  rs.forEach(r=>ord.forEach(m=>{if(m in r)have.add(m)}));
  return ord.filter(m=>have.has(m));
}

// ── Column visibility toggles ────────────────────────────────────────
function renderTextColToggles(){
  // called once at init for text cols (static)
  const c=document.getElementById('tcols');
  const textHtml=TEXT_COLS.map(({key,label})=>{
    const on=visTextCols.has(key)?'on':'';
    return `<label class="tcol-lbl ${on}" id="tl-${key}"><input type="checkbox" value="${key}" ${on?'checked':''} onchange="onTcol(this)"> ${label}</label>`;
  }).join('');
  c.innerHTML=textHtml+'<span id="meth-col-toggles"></span>';
}

function renderMethodColToggles(){
  // called after task load; appended after text col toggles
  const span=document.getElementById('meth-col-toggles');
  if(!span)return;
  span.innerHTML=(methodCols.length?'<span style="color:var(--dim);font-size:10px;margin:0 4px">|</span>':'')+
    methodCols.map(m=>{
      const on=visMethodCols.has(m)?'on':'';
      return `<label class="tcol-lbl ${on}" id="tl-m-${m}"><input type="checkbox" value="${m}" ${on?'checked':''} onchange="onMcol(this)"> ${sn(m)}</label>`;
    }).join('');
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

// ── Method checkboxes ────────────────────────────────────────────────
const DEFAULT_CMP_METHODS=new Set(['our_ao','llm_monitor_flash']);
function renderMcbs(){
  const c=document.getElementById('mcbs');
  c.innerHTML=methodCols.map(m=>{const on=DEFAULT_CMP_METHODS.has(m)?'on':'';const chk=DEFAULT_CMP_METHODS.has(m)?'checked':'';return `<label class="mcb-lbl ${on}" id="ml-${m}"><input type="checkbox" value="${m}" ${chk} onchange="onMcb(this)"> ${sn(m)}</label>`;}).join('');
  document.getElementById('cmp-row').style.display='flex';
}

function onMcb(cb){
  document.getElementById('ml-'+cb.value)?.classList.toggle('on',cb.checked);
  rerender();_tsave();
}
function mcbAll(){document.querySelectorAll('#mcbs input').forEach(cb=>{cb.checked=true;document.getElementById('ml-'+cb.value)?.classList.add('on')});rerender();_tsave();}
function mcbNone(){document.querySelectorAll('#mcbs input').forEach(cb=>{cb.checked=false;document.getElementById('ml-'+cb.value)?.classList.remove('on')});rerender();_tsave();}
function selMethods(){const c=Array.from(document.querySelectorAll('#mcbs input:checked')).map(x=>x.value);return c.length?c:methodCols}

// ── Disagreement (stdev of LLM judge scores for selected methods) ────
function dis(r){
  const ms=selMethods(), sc=r.llm_comparative_score||{};
  const vs=ms.filter(m=>m in sc).map(m=>typeof sc[m]==='object'?sc[m].score:sc[m]);
  if(vs.length<2)return 0;
  const mu=vs.reduce((a,b)=>a+b,0)/vs.length;
  return Math.sqrt(vs.reduce((s,v)=>s+(v-mu)**2,0)/(vs.length-1));
}

// ── Render ────────────────────────────────────────────────────────────
function rerender(){
  const q=document.getElementById('search').value.toLowerCase();
  filterQ=q;
  const filtered=q?recs.filter(r=>JSON.stringify(r).toLowerCase().includes(q)):recs;
  const sorted=[...filtered].sort((a,b)=>{
    let av,bv;
    if(sortKey==='dis'){av=dis(a);bv=dis(b)}
    else if(sortKey==='agree'){av=a._pred_agreement;bv=b._pred_agreement}
    else if(sortKey==='idx'){av=recs.indexOf(a);bv=recs.indexOf(b)}
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
    if(expanded.has(oi))h.push(detailHtml(r,cols.length));
  });
  h.push('</tbody></table>');
  document.getElementById('wrap').innerHTML=h.join('');
}

function buildCols(){
  const cols=[
    {id:'idx', cc:'c-idx', lbl:'eval_id', sk:'idx', defaultW:140},
    {id:'tog', cc:'c-tog', lbl:'',     sk:null,     defaultW:22},
    {id:'dis', cc:'c-dis', lbl:'dis',  sk:'dis',    defaultW:58},
    {id:'ag',  cc:'c-ag',  lbl:'agr',  sk:'agree',  defaultW:50},
  ];
  const cotLabel=recs.length&&recs[0]._cot_field_label?recs[0]._cot_field_label:null;
  TEXT_COLS.forEach(({key,label})=>{
    if(!visTextCols.has(key))return;
    let lbl=label;
    if(key==='cot_field'&&cotLabel) lbl=`${label} <span style="color:var(--orange);font-size:9px">(${cotLabel})</span>`;
    cols.push({id:'t:'+key, cc:'c-txt', lbl, sk:'t:'+key, defaultW:180});
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
  const d=dis(r), dcl=d>.2?'dh':d>.1?'dm':'dl';
  const ag=r._pred_agreement||0, acl=ag>.85?'ah':ag>.6?'am':'al';
  const sc=r.llm_comparative_score||{};
  let h=`<tr class="${isX?'xrow':''}" id="row-${oi}">`;
  h+=`<td class="c-idx" title="${esc(String(r.example_id??''))}">${esc(String(r.example_id??''))}</td>`;
  h+=`<td class="c-tog" onclick="tog(${oi})">${isX?'▼':'▶'}</td>`;
  h+=`<td class="c-dis ${dcl}" title="${d.toFixed(3)}">${d.toFixed(3)}</td>`;
  h+=`<td class="c-ag  ${acl}" title="${(ag*100).toFixed(0)}%">${(ag*100).toFixed(0)}%</td>`;
  // text cols
  TEXT_COLS.forEach(({key})=>{
    if(!visTextCols.has(key))return;
    const v=String(r[key]||'');
    h+=`<td class="c-txt" title="${esc(v)}">${esc(v)}</td>`;
  });
  // method predictions with inline abs/cmp badges
  methodCols.filter(m=>visMethodCols.has(m)).forEach(m=>{
    const v=r[m]!==undefined?String(r[m]):'—';
    const av=r._abs_scores?.[m];
    const cv_raw=sc[m]; const cv=cv_raw!==undefined?(typeof cv_raw==='object'?cv_raw.score:cv_raw):undefined;
    const acl=av!==undefined?(av>=0.5?'sh':'sl'):'pu';
    const ccl=cv!==undefined?(cv>=0.6?'cph':cv>=0.4?'cpm':'cpl'):'';
    const absVal=av!==undefined?(absType==='f1'?av.toFixed(2):av.toFixed(0)):'';
    const absBdg=av!==undefined?`<span class="bdg ${acl}">abs:${absVal}</span>`:'';
    const cmpBdg=cv!==undefined?`<span class="bdg ${ccl}">cmp:${cv.toFixed(2)}</span>`:'';
    const tip=`abs(${absType}):${absVal} cmp:${cv!==undefined?cv.toFixed(2):'—'}\n${esc(v)}`;
    h+=`<td class="c-meth" title="${tip}">${absBdg}${cmpBdg}<span class="pred-txt">${esc(v)}</span></td>`;
  });
  h+='</tr>';
  return h;
}

function detailHtml(r,colspan){
  const sc=r.llm_comparative_score||{};
  const allVals=Object.values(sc).map(e=>typeof e==='object'?e.score:e);
  const maxV=Math.max(...allVals,.001);
  let h=`<tr class="drow"><td colspan="${colspan}"><div class="dbox">`;

  // Text fields grid
  const txts=TEXT_COLS.filter(({key})=>r[key]);
  if(txts.length){
    h+=`<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;margin-bottom:8px">`;
    txts.forEach(({key,label})=>{
      const v=esc(String(r[key]));
      const lbl=(key==='cot_field'&&r._cot_field_label)?`${label} <span style="color:var(--orange);font-size:9px">(${r._cot_field_label})</span>`:label;
      h+=`<div><div style="font-size:9px;text-transform:uppercase;color:var(--dim);margin-bottom:2px">${lbl}</div>
          <div style="background:var(--bg);border:1px solid var(--border);border-radius:2px;padding:5px 7px;font-size:11px;line-height:1.5;max-height:180px;overflow-y:auto;white-space:pre-wrap;word-break:break-word">${v}</div></div>`;
    });
    h+=`</div>`;
  }

  // Methods table — all methods with predictions; LCS score+reason if available
  const hasPreds=methodCols.some(m=>r[m]!==undefined);
  if(hasPreds){
    const hasLcs=methodCols.some(m=>m in sc);
    const dcols=[{id:'method',lbl:'method',dw:120},...(hasLcs?[{id:'score',lbl:'score (cmp)',dw:60},{id:'bar',lbl:'',dw:90}]:[]),{id:'pred',lbl:'prediction',dw:340},...(hasLcs?[{id:'reason',lbl:'cmp reason (gemini-flash)',dw:180}]:[]),{id:'btn',lbl:'',dw:80}];
    const cg=`<colgroup>${dcols.map(c=>`<col data-dc="${c.id}" style="width:${dtblColWidths[c.id]??c.dw}px">`).join('')}</colgroup>`;
    const hdr=dcols.map(c=>`<th>${c.lbl}<div class="rh" onpointerdown="dsr(event,'${c.id}')"></div></th>`).join('');
    h+=`<div class="dtbl-wrap"><table class="dtbl">${cg}<tr>${hdr}</tr>`;
    methodCols.forEach(m=>{
      if(r[m]===undefined&&!(m in sc))return;
      const e=sc[m];
      const scoreCell=hasLcs?(e!==undefined?(()=>{
        const v=typeof e==='object'?e.score:e;
        const cl=v>=.7?'sh':v>=.4?'sm_':'sl';
        const bc=v>=.7?'#7ec480':v>=.4?'#cc8a40':'#c97b7b';
        const bw=Math.round(v/maxV*80);
        return `<td><span class="sv ${cl}">${v.toFixed(3)}</span></td><td><span class="sbar" style="width:${bw}px;background:${bc}"></span></td>`;
      })():`<td>—</td><td></td>`):'';
      const rsn=e!==undefined&&typeof e==='object'?(e.reason||''):'';
      const reasonCell=hasLcs?`<td class="rsn">${esc(rsn)}</td>`:'';
      const pred=r[m]!==undefined?esc(String(r[m])):'—';
      const saeCid=`sae-${esc(String(r.example_id??''))}`;
      const saeBtn=m==='sae_probe'?`<button class="sae-btn" id="${saeCid}-btn" onclick="toggleSaeInline('${currentTask}','${esc(String(r.example_id??''))}','${saeCid}')">▶ features</button>`:'';
      h+=`<tr><td style="white-space:nowrap">${m}</td>${scoreCell}<td class="pred-cell">${pred}${m==='sae_probe'?`<div id="${saeCid}"></div>`:''}</td>${reasonCell}<td>${saeBtn}</td></tr>`;
    });
    h+=`</table></div>`;
  }
  h+=`</div></td></tr>`;
  return h;
}

// ── Toggle / sort ─────────────────────────────────────────────────────
function tog(oi){expanded.has(oi)?expanded.delete(oi):expanded.add(oi);rerender();_tsave();}
function setSort(k){sortDir=(sortKey===k)?-sortDir:-1;sortKey=k;['dis','agree','idx'].forEach(x=>document.getElementById('sb-'+x)?.classList.toggle('on',x===k));rerender();_gsave();}

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
function showSaeFeatures(task, exampleId, btn){
  btn.textContent='…';
  fetch(`/api/sae_features/${encodeURIComponent(task)}/${encodeURIComponent(exampleId)}`)
    .then(r=>r.json())
    .then(d=>{
      btn.textContent='features';
      const text=d.error?d.error:(d.feature_desc||'(no feature_desc)');
      const prompt=d.full_prompt?`\n\n--- FULL PROMPT ---\n${d.full_prompt}`:'';
      const overlay=document.createElement('div');
      overlay.className='sae-modal';
      overlay.innerHTML=`<div class="sae-modal-box"><button class="sae-modal-close" onclick="this.closest('.sae-modal').remove()">✕</button><b>SAE features — ${esc(exampleId)}</b>\n\n${esc(text+prompt)}</div>`;
      document.body.appendChild(overlay);
      overlay.addEventListener('click',e=>{if(e.target===overlay)overlay.remove()});
    })
    .catch(e=>{btn.textContent='features';alert('Error: '+e)});
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
function sn(m){return m.replace('llm_monitor_','').replace(/_/g,' ')}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

function thMethodLbl(m){
  const ms=methodScores[m];
  const tr=ms?.trained?`<span class="th-tr" title="trained on this task">†</span>`:'';
  let vbars='';
  if(ms){
    const abs=ms.primary_score??0;
    const cmp=ms.comparative_score;
    const vbar=(lbl,val,cls)=>`<div class="th-vbg"><div class="th-vbt"><div class="th-vbf ${cls}" style="height:${(val*100).toFixed(0)}%" title="${lbl}: ${val.toFixed(2)}"></div></div><span class="th-vbl">${lbl}</span></div>`;
    vbars=`<div class="th-vbars">${vbar('abs('+absType+')',abs,'th-ba')}${cmp!=null?vbar('cmp',cmp,'th-bc'):''}</div>`;
  }
  const modelLbl=ms?.model?`<div class="th-model" title="${esc(ms.model)}">${esc(ms.model.split('/').pop())}</div>`:'';
  return `<div class="th-mc"><div class="th-mn">${sn(m)}${tr}</div>${modelLbl}${vbars}</div>`;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8788, debug=False)
