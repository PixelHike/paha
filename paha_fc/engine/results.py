"""Run records: per-run summaries, per-task traces and cross-seed aggregation."""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..config import Config

RUN_FIELDS = ("seed", "final_acc", "forgetting", "elapsed_s", "n_clients", "participation",
              "dataset", "num_tasks", "alpha", "ablation", "tag")


def run_signature(cfg: Config, ablation: str = "full") -> str:
    """Filesystem-safe label of a comparison cell; seeds accumulate in one folder."""
    alpha = "iid" if cfg.alpha <= 0 else f"a{cfg.alpha}"
    return "_".join([
        cfg.dataset, f"T{cfg.num_tasks}", f"N{cfg.n_clients}", f"p{cfg.participation:g}",
        alpha, ablation, cfg.tag or "untagged",
    ]).replace(".", "p")


class ResultStore:
    """Writes ``config.json``, ``runs.jsonl``, ``tasks.jsonl`` and ``summary.csv``."""

    def __init__(self, cfg: Config, ablation: str = "full", subdir: Optional[str] = None,
                 enabled: bool = True):
        self.cfg = cfg
        self.ablation = ablation
        self.enabled = enabled
        self.dir = os.path.join(cfg.out_dir, subdir or run_signature(cfg, ablation))
        self.runs: List[Dict] = []
        if not self.enabled:
            return
        os.makedirs(self.dir, exist_ok=True)
        self._write_config(cfg.to_dict())

    def _write_config(self, payload: Dict) -> None:
        """``config.json`` is a plain ``Config`` dump, reusable via ``--config``."""
        if not self.enabled:
            return
        with open(os.path.join(self.dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    # ------------------------------------------------------------- appending
    def _append_jsonl(self, filename: str, record: Dict) -> None:
        """Append one line, stamped with the run bookkeeping."""
        if not self.enabled:
            return
        line = {"seed": self.cfg.seed, "tag": self.cfg.tag, "ablation": self.ablation,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        line.update(record)
        with open(os.path.join(self.dir, filename), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=float) + "\n")

    def append_task(self, task_record: Dict) -> None:
        """Per-task diagnostics line, stamped by :meth:`_append_jsonl`."""
        self._append_jsonl("tasks.jsonl", task_record)

    def record_run(self, result, seed: Optional[int] = None,
                   extra: Optional[Dict] = None) -> Dict:
        """Record one run and refresh ``config.json`` with the effective settings."""
        entry = {
            "seed": int(seed if seed is not None else self.cfg.seed),
            "final_acc": float(result.final_acc),
            "forgetting": float(result.forgetting),
            "elapsed_s": float(result.elapsed_s),
            "n_clients": self.cfg.n_clients,
            "participation": self.cfg.participation,
            "dataset": self.cfg.dataset,
            "num_tasks": self.cfg.num_tasks,
            "alpha": self.cfg.alpha,
            "acc_history": result.acc_history,
            "task_splits": result.task_splits,
        }
        if extra:
            entry.update(extra)
        self.runs.append(entry)
        self._append_jsonl("runs.jsonl", entry)
        effective = getattr(result, "config", None)
        if effective:
            self._write_config(effective)
        return entry

    # ------------------------------------------------------------ summary
    def recorded_runs(self) -> List[Dict]:
        """Runs in this process merged with what earlier runs left on disk."""
        merged = {(int(r.get("seed", -1)), r.get("ablation", ""), r.get("tag", "")): r
                  for r in self.runs}
        for record in load_runs(os.path.join(self.dir, "runs.jsonl")):
            merged.setdefault((int(record.get("seed", -1)), record.get("ablation", ""),
                               record.get("tag", "")), record)
        return list(merged.values())

    def summarize(self) -> Dict[str, Dict[str, float]]:
        runs = self.recorded_runs()
        stats = aggregate_runs(runs)
        if self.enabled and runs:
            _write_csv(os.path.join(self.dir, "summary.csv"), stats)
            stats["runs"] = {"count": float(len(runs))}
        return stats

    def write_matrix(self, acc_matrix: Iterable[Dict]) -> None:
        """Cross-configuration table used by the sweep driver."""
        rows = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))}
                for row in acc_matrix]
        if self.enabled and rows:
            _write_csv(os.path.join(self.dir, "matrix.csv"), rows)


def _write_csv(path: str, rows) -> None:
    if isinstance(rows, dict):
        rows = [{"metric": k, "mean": v.get("mean"), "std": v.get("std"),
                 "min": v.get("min"), "max": v.get("max")} for k, v in rows.items()]
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def aggregate_runs(runs: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Mean/std/min/max of the headline metrics across seeds."""
    out: Dict[str, Dict[str, float]] = {}
    for metric in ("final_acc", "forgetting", "elapsed_s"):
        values = [float(r[metric]) for r in runs if metric in r]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[metric] = {"mean": float(arr.mean()), "std": float(arr.std()),
                       "min": float(arr.min()), "max": float(arr.max()),
                       "n": float(len(arr))}
    return out


def load_runs(path: str) -> List[Dict]:
    """Read a ``runs.jsonl`` file back into memory."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
