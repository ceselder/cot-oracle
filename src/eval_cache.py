"""SQLite-backed modular eval cache.

Each (task × method × item) cell is independently addressable, so adding
one baseline never invalidates results for other methods.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = "v2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    checkpoint TEXT NOT NULL,
    n_examples INTEGER NOT NULL,
    position_mode TEXT NOT NULL,
    layers TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    item_idx INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    target_response TEXT NOT NULL,
    PRIMARY KEY (run_id, task_name, item_idx)
);

CREATE TABLE IF NOT EXISTS method_results (
    run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    method_name TEXT NOT NULL,
    primary_score REAL,
    bootstrap_std REAL,
    primary_metric TEXT NOT NULL,
    n INTEGER NOT NULL,
    extra_json TEXT,
    deps_hash TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, task_name, method_name)
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    method_name TEXT NOT NULL,
    item_idx INTEGER NOT NULL,
    item_id TEXT NOT NULL,
    prediction TEXT,
    score REAL,
    prompt TEXT,
    scorer_response TEXT,
    PRIMARY KEY (run_id, task_name, method_name, item_idx)
);

CREATE TABLE IF NOT EXISTS comparative_scores (
    run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    item_idx INTEGER NOT NULL,
    best_method TEXT NOT NULL,
    justification TEXT,
    method_scores_json TEXT,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, task_name, item_idx)
);
"""


def _sha256_short(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


class EvalCache:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Migrations for existing DBs
        for col in ["prompt TEXT", "scorer_response TEXT"]:
            try:
                self._conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        # v2: comparative_scores table
        try:
            self._conn.execute("SELECT 1 FROM comparative_scores LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS comparative_scores (
                    run_id TEXT NOT NULL, task_name TEXT NOT NULL, item_idx INTEGER NOT NULL,
                    best_method TEXT NOT NULL, justification TEXT, method_scores_json TEXT,
                    computed_at TEXT NOT NULL, PRIMARY KEY (run_id, task_name, item_idx)
                );
            """)

    def close(self):
        self._conn.close()

    # ── Run management ──

    def get_or_create_run(self, checkpoint: str, n_examples: int,
                          position_mode: str, layers: list[int],
                          schema_version: str = SCHEMA_VERSION) -> str:
        layers_json = json.dumps(sorted(layers))
        run_id = _sha256_short(checkpoint, n_examples, position_mode, layers_json, schema_version)
        row = self._conn.execute("SELECT run_id FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO eval_runs (run_id, checkpoint, n_examples, position_mode, layers, schema_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, checkpoint, n_examples, position_mode, layers_json, schema_version, datetime.now().isoformat()),
            )
            self._conn.commit()
        return run_id

    # ── deps_hash ──

    def method_deps_hash(self, run_id: str, task_name: str, method_name: str, method_config: dict) -> str:
        return _sha256_short(run_id, task_name, method_name, json.dumps(method_config, sort_keys=True))

    # ── Query ──

    # Incomplete markers: primary_metric values that indicate incomplete eval
    FAILURE_METRICS = {"error", "failed"}      # hard failures — need full rerun
    INCOMPLETE_METRICS = {"error", "failed", "unscored"}  # any non-final state

    def has_method(self, run_id: str, task_name: str, method_name: str, expected_deps_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT deps_hash, primary_metric FROM method_results WHERE run_id = ? AND task_name = ? AND method_name = ?",
            (run_id, task_name, method_name),
        ).fetchone()
        if row is None:
            return False
        # Incomplete results don't count — they should be retried
        if row[1] in self.INCOMPLETE_METRICS:
            return False
        return row[0] == expected_deps_hash

    def get_method_status(self, run_id: str, task_name: str, method_name: str) -> str | None:
        """Return primary_metric for this method, or None if not present."""
        row = self._conn.execute(
            "SELECT primary_metric FROM method_results WHERE run_id = ? AND task_name = ? AND method_name = ?",
            (run_id, task_name, method_name),
        ).fetchone()
        return row[0] if row else None

    def get_failed_methods(self, run_id: str) -> list[tuple[str, str, str]]:
        """Return [(task_name, method_name, reason)] for all failed/unscored evals."""
        rows = self._conn.execute(
            "SELECT task_name, method_name, primary_metric, extra_json FROM method_results WHERE run_id = ? AND primary_metric IN ('error', 'failed', 'unscored')",
            (run_id,),
        ).fetchall()
        result = []
        for task, method, metric, extra_json in rows:
            extra = json.loads(extra_json) if extra_json else {}
            if metric == "unscored":
                reason = "unscored (predictions cached, scoring failed)"
            else:
                reason = extra.get("error", extra.get("reason", "unknown"))
            result.append((task, method, reason))
        return result

    def get_unscored_methods(self, run_id: str) -> list[tuple[str, str]]:
        """Return [(task_name, method_name)] for methods with predictions but no scores."""
        rows = self._conn.execute(
            "SELECT task_name, method_name FROM method_results WHERE run_id = ? AND primary_metric = 'unscored'",
            (run_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def has_predictions(self, run_id: str, task_name: str, method_name: str) -> bool:
        """Check if any predictions exist for this method (regardless of scoring status)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE run_id = ? AND task_name = ? AND method_name = ?",
            (run_id, task_name, method_name),
        ).fetchone()
        return row[0] > 0

    def get_completed_item_indices(self, run_id: str, task_name: str, method_name: str) -> set[int]:
        rows = self._conn.execute(
            "SELECT item_idx FROM predictions WHERE run_id = ? AND task_name = ? AND method_name = ?",
            (run_id, task_name, method_name),
        ).fetchall()
        return {r[0] for r in rows}

    def get_predictions(self, run_id: str, task_name: str, method_name: str) -> dict[int, dict]:
        """Return {item_idx: {"prediction": str, "score": float|None, "prompt": str|None, "scorer_response": str|None}}."""
        rows = self._conn.execute(
            "SELECT item_idx, prediction, score, prompt, scorer_response FROM predictions WHERE run_id = ? AND task_name = ? AND method_name = ?",
            (run_id, task_name, method_name),
        ).fetchall()
        return {r[0]: {"prediction": r[1], "score": r[2], "prompt": r[3], "scorer_response": r[4]} for r in rows}

    # ── Store ──

    def store_items(self, run_id: str, task_name: str, items: list[dict]):
        self._conn.executemany(
            "INSERT OR REPLACE INTO items (run_id, task_name, item_idx, item_id, target_response) VALUES (?, ?, ?, ?, ?)",
            [(run_id, task_name, it["item_idx"], it["item_id"], it["target_response"]) for it in items],
        )
        self._conn.commit()

    def store_predictions(self, run_id: str, task_name: str, method_name: str, preds: list[dict]):
        self._conn.executemany(
            "INSERT OR REPLACE INTO predictions (run_id, task_name, method_name, item_idx, item_id, prediction, score, prompt, scorer_response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(run_id, task_name, method_name, p["item_idx"], p["item_id"], p.get("prediction"), p.get("score"), p.get("prompt"), p.get("scorer_response")) for p in preds],
        )
        self._conn.commit()

    def store_method_result(self, run_id: str, task_name: str, method_name: str, result: dict, deps_hash: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO method_results (run_id, task_name, method_name, primary_score, bootstrap_std, primary_metric, n, extra_json, deps_hash, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, task_name, method_name, result["primary_score"], result.get("bootstrap_std", 0.0),
             result["primary_metric"], result["n"], json.dumps(result.get("extra", {})),
             deps_hash, datetime.now().isoformat()),
        )
        self._conn.commit()

    # ── Comparative scores ──

    def store_comparative_scores(self, run_id: str, task_name: str, scores: list[dict]):
        """Store per-item comparative scores. Each dict: {item_idx, best_method, justification, method_scores}."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO comparative_scores (run_id, task_name, item_idx, best_method, justification, method_scores_json, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(run_id, task_name, s["item_idx"], s["best_method"], s.get("justification", ""),
              json.dumps(s.get("method_scores", {})), datetime.now().isoformat()) for s in scores],
        )
        self._conn.commit()

    def get_comparative_scores(self, run_id: str, task_name: str) -> dict[int, dict]:
        """Return {item_idx: {best_method, justification, method_scores}}."""
        rows = self._conn.execute(
            "SELECT item_idx, best_method, justification, method_scores_json FROM comparative_scores WHERE run_id = ? AND task_name = ?",
            (run_id, task_name),
        ).fetchall()
        return {r[0]: {"best_method": r[1], "justification": r[2], "method_scores": json.loads(r[3]) if r[3] else {}} for r in rows}

    def has_comparative_scores(self, run_id: str, task_name: str) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM comparative_scores WHERE run_id = ? AND task_name = ?",
            (run_id, task_name),
        ).fetchone()
        return row[0] > 0

    # ── Delete ──

    def delete_method(self, run_id: str, task_name: str, method_name: str):
        self._conn.execute("DELETE FROM method_results WHERE run_id = ? AND task_name = ? AND method_name = ?", (run_id, task_name, method_name))
        self._conn.execute("DELETE FROM predictions WHERE run_id = ? AND task_name = ? AND method_name = ?", (run_id, task_name, method_name))
        self._conn.commit()

    def delete_task(self, run_id: str, task_name: str):
        self._conn.execute("DELETE FROM method_results WHERE run_id = ? AND task_name = ?", (run_id, task_name))
        self._conn.execute("DELETE FROM predictions WHERE run_id = ? AND task_name = ?", (run_id, task_name))
        self._conn.execute("DELETE FROM items WHERE run_id = ? AND task_name = ?", (run_id, task_name))
        self._conn.commit()

    # ── Export ──

    def export_task_json(self, run_id: str, task_name: str) -> dict:
        """Build complete JSON for one task from DB state."""
        # Items
        item_rows = self._conn.execute(
            "SELECT item_idx, item_id, target_response FROM items WHERE run_id = ? AND task_name = ? ORDER BY item_idx",
            (run_id, task_name),
        ).fetchall()
        items_by_idx = {r[0]: {"id": r[1], "target_response": r[2], "methods": {}} for r in item_rows}

        # Methods
        method_rows = self._conn.execute(
            "SELECT method_name, primary_score, bootstrap_std, primary_metric, n, extra_json, deps_hash, computed_at FROM method_results WHERE run_id = ? AND task_name = ?",
            (run_id, task_name),
        ).fetchall()
        methods = {}
        for mr in method_rows:
            extra = json.loads(mr[5]) if mr[5] else {}
            methods[mr[0]] = {
                "primary_score": mr[1],
                "bootstrap_std": mr[2],
                "primary_metric": mr[3],
                "n": mr[4],
                **extra,
            }

        # Predictions
        pred_rows = self._conn.execute(
            "SELECT method_name, item_idx, prediction, score, prompt, scorer_response FROM predictions WHERE run_id = ? AND task_name = ?",
            (run_id, task_name),
        ).fetchall()
        for pr in pred_rows:
            method_name, item_idx, prediction, score, prompt, scorer_response = pr
            if item_idx in items_by_idx:
                entry = {"prediction": prediction, "score": score}
                if prompt:
                    entry["prompt"] = prompt
                if scorer_response:
                    entry["scorer_response"] = scorer_response
                items_by_idx[item_idx]["methods"][method_name] = entry

        # Run info
        run_row = self._conn.execute("SELECT checkpoint, n_examples, position_mode, layers FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()

        return {
            "_cache_meta": {
                "run_id": run_id,
                "checkpoint": run_row[0] if run_row else "",
                "n_examples": run_row[1] if run_row else 0,
                "position_mode": run_row[2] if run_row else "",
                "layers": json.loads(run_row[3]) if run_row else [],
            },
            "task": task_name,
            "n_examples": run_row[1] if run_row else 0,
            "timestamp": datetime.now().isoformat(),
            "methods": methods,
            "items": [items_by_idx[k] for k in sorted(items_by_idx)],
        }

    def export_all_json(self, run_id: str, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = [r[0] for r in self._conn.execute("SELECT DISTINCT task_name FROM method_results WHERE run_id = ?", (run_id,)).fetchall()]
        for task_name in tasks:
            data = self.export_task_json(run_id, task_name)
            (output_dir / f"{task_name}.json").write_text(json.dumps(data, indent=2, default=str))

    def get_all_method_results(self, run_id: str) -> dict[str, dict[str, dict]]:
        """Returns {task_name: {method_name: {primary_score, bootstrap_std, ...}}}."""
        rows = self._conn.execute(
            "SELECT task_name, method_name, primary_score, bootstrap_std, primary_metric, n, extra_json FROM method_results WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        result: dict[str, dict[str, dict]] = {}
        for r in rows:
            task, method = r[0], r[1]
            extra = json.loads(r[6]) if r[6] else {}
            result.setdefault(task, {})[method] = {
                "primary_score": r[2],
                "bootstrap_std": r[3],
                "primary_metric": r[4],
                "n": r[5],
                **extra,
            }
        return result
