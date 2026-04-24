#!/usr/bin/env python3
"""Generate consolidated run metrics CSV files.

This script prefers structured evaluation JSON files written by val_simple.py
and falls back to parsing SLURM logs when JSON is unavailable.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _float_or_empty(value: object) -> str:
    if value is None:
        return ""
    try:
        return str(float(value))
    except Exception:
        return ""


def _parse_best_from_log(text: str) -> tuple[str, str]:
    best = re.findall(r"\*\*\* Best metric: ([0-9.]+) \(epoch ([0-9]+)\)", text)
    if not best:
        return "", ""
    return best[-1][0], best[-1][1]


def _parse_checkpoint_from_log(text: str) -> str:
    ckpt = re.findall(r"Current checkpoints:\s*\n\s*\('([^']+)',\s*([0-9.]+)\)", text)
    if not ckpt:
        return ""
    return ckpt[-1][0]


def _load_eval_json(metrics_json: Path) -> Dict[str, str]:
    payload = json.loads(metrics_json.read_text())
    return {
        "best_top1": _float_or_empty(payload.get("top1")),
        "eval_loss": _float_or_empty(payload.get("loss")),
        "precision": _float_or_empty(payload.get("precision")),
        "recall": _float_or_empty(payload.get("recall")),
        "f1": _float_or_empty(payload.get("f1")),
    }


def _infer_eval_json_from_log(text: str) -> Optional[Path]:
    saved = re.findall(r"Saved metrics JSON to\s+(.+)", text)
    if saved:
        return Path(saved[-1].strip())

    # Fallback when subprocess error contains command args.
    cmd_json = re.findall(r"--metrics-json', '([^']+)'", text)
    if cmd_json:
        return Path(cmd_json[-1].strip())

    return None


def _parse_metrics_payload_from_text(text: str) -> Dict[str, str]:
    # Prefer explicit JSON metrics block when available.
    json_blocks = re.findall(r"JSON_METRICS_START\s*(\{.*?\})\s*JSON_METRICS_END", text, flags=re.S)
    if json_blocks:
        try:
            data = json.loads(json_blocks[-1])
            return {
                "best_top1": _float_or_empty(data.get("top1")),
                "eval_loss": _float_or_empty(data.get("loss")),
                "precision": _float_or_empty(data.get("precision")),
                "recall": _float_or_empty(data.get("recall")),
                "f1": _float_or_empty(data.get("f1")),
            }
        except Exception:
            pass

    payloads = re.findall(r"OrderedDict\((\{.*?\})\)", text, flags=re.S)
    if payloads:
        try:
            data = ast.literal_eval(payloads[-1])
            return {
                "best_top1": _float_or_empty(data.get("top1")),
                "eval_loss": _float_or_empty(data.get("loss")),
                "precision": _float_or_empty(data.get("precision")),
                "recall": _float_or_empty(data.get("recall")),
                "f1": _float_or_empty(data.get("f1")),
            }
        except Exception:
            pass

    return {}


def build_rows(repo: Path) -> List[Dict[str, str]]:
    slurm = repo / "slurm_logs"
    rows: List[Dict[str, str]] = []

    for err in sorted(slurm.glob("training_*.err")):
        text = err.read_text(errors="ignore")
        job_id = err.stem.split("_")[-1]
        best_top1, best_epoch = _parse_best_from_log(text)
        checkpoint_path = _parse_checkpoint_from_log(text)
        experiment = Path(checkpoint_path).parent.name if checkpoint_path else ""

        rows.append(
            {
                "source": "training.slurm",
                "job_id": job_id,
                "experiment": experiment,
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "eval_loss": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "checkpoint_path": checkpoint_path,
                "log_file": str(err.relative_to(repo)),
            }
        )

    for err in sorted(slurm.glob("launch_*.err")):
        text = err.read_text(errors="ignore")
        stem = err.stem
        job_id = stem.split("_")[-1]
        source = (
            "launch_experiments.slurm"
            if "launch_experiments_" in stem
            else "launch_experiments_ablation.slurm"
        )
        best_top1, best_epoch = _parse_best_from_log(text)
        checkpoint_path = _parse_checkpoint_from_log(text)
        experiment = Path(checkpoint_path).parent.name if checkpoint_path else ""

        rows.append(
            {
                "source": source,
                "job_id": job_id,
                "experiment": experiment,
                "best_top1": best_top1,
                "best_epoch": best_epoch,
                "eval_loss": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "checkpoint_path": checkpoint_path,
                "log_file": str(err.relative_to(repo)),
            }
        )

    # Evaluation rows: prefer metrics JSON, fallback to JSON_METRICS/OrderedDict in logs.
    for log in sorted(list(slurm.glob("evaluation_*.out")) + list(slurm.glob("evaluation_*.err"))):
        text = log.read_text(errors="ignore")
        job_id = log.stem.split("_")[-1]
        metrics_json = _infer_eval_json_from_log(text)

        row = {
            "source": "evaluation.slurm",
            "job_id": job_id,
            "experiment": "",
            "best_top1": "",
            "best_epoch": "",
            "eval_loss": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "checkpoint_path": "",
            "log_file": str(log.relative_to(repo)),
        }

        if metrics_json and metrics_json.exists():
            row.update(_load_eval_json(metrics_json))
            row["experiment"] = metrics_json.parent.name
            row["checkpoint_path"] = str((metrics_json.parent / "model_best.pth.tar"))
            rows.append(row)
            continue

        parsed = _parse_metrics_payload_from_text(text)
        if parsed:
            row.update(parsed)
            rows.append(row)

    # Testing rows: parse eval-style payloads from testing logs.
    for log in sorted(list(slurm.glob("testing_*.out")) + list(slurm.glob("testing_*.err"))):
        text = log.read_text(errors="ignore")
        job_id = log.stem.split("_")[-1]

        row = {
            "source": "testing.slurm",
            "job_id": job_id,
            "experiment": "",
            "best_top1": "",
            "best_epoch": "",
            "eval_loss": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "checkpoint_path": "",
            "log_file": str(log.relative_to(repo)),
        }

        checkpoint_match = re.findall(r"Checkpoint\s*:\s*(.+)", text)
        if checkpoint_match:
            ckpt = checkpoint_match[-1].strip()
            row["checkpoint_path"] = ckpt
            row["experiment"] = Path(ckpt).parent.name

        parsed = _parse_metrics_payload_from_text(text)
        if parsed:
            row.update(parsed)
            rows.append(row)

    # Ensure every evaluation_metrics.json has representation even if no evaluation log captured it.
    seen_eval_experiments = {
        r["experiment"]
        for r in rows
        if r["source"] == "evaluation.slurm" and r["experiment"]
    }

    for metrics_json in sorted((repo / "weights").glob("*/evaluation_metrics.json")):
        experiment = metrics_json.parent.name
        if experiment in seen_eval_experiments:
            continue
        row = {
            "source": "evaluation.json",
            "job_id": "",
            "experiment": experiment,
            "best_top1": "",
            "best_epoch": "",
            "eval_loss": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "checkpoint_path": str(metrics_json.parent / "model_best.pth.tar"),
            "log_file": str(metrics_json.relative_to(repo)),
        }
        row.update(_load_eval_json(metrics_json))
        rows.append(row)

    rows.sort(key=lambda r: (int(r["job_id"]) if r["job_id"].isdigit() else 0, r["source"]))
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "source",
        "job_id",
        "experiment",
        "best_top1",
        "best_epoch",
        "eval_loss",
        "precision",
        "recall",
        "f1",
        "checkpoint_path",
        "log_file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_success_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    successful = []
    for row in rows:
        has_training_metric = bool((row.get("best_top1") or "").strip())
        has_eval_metric = bool((row.get("eval_loss") or "").strip())
        if not (has_training_metric or has_eval_metric):
            continue

        try:
            sort_score = float(row.get("best_top1") or -1.0)
        except Exception:
            sort_score = -1.0

        clean = dict(row)
        clean["_sort_score"] = str(sort_score)
        successful.append(clean)

    successful.sort(
        key=lambda r: (
            float(r["_sort_score"]),
            int(r["job_id"]) if str(r["job_id"]).isdigit() else 0,
        ),
        reverse=True,
    )

    for row in successful:
        row.pop("_sort_score", None)

    write_csv(path, successful)


def write_best_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    best_by_group: Dict[str, Dict[str, str]] = {}

    for row in rows:
        score_raw = (row.get("best_top1") or "").strip()
        if not score_raw:
            continue

        try:
            score = float(score_raw)
        except Exception:
            continue

        key = row.get("source", "")
        previous = best_by_group.get(key)
        if previous is None:
            best_by_group[key] = dict(row)
            continue

        try:
            prev_score = float(previous.get("best_top1") or "-1")
        except Exception:
            prev_score = -1.0

        if score > prev_score:
            best_by_group[key] = dict(row)
            continue

        if score == prev_score:
            prev_job = int(previous.get("job_id") or 0) if str(previous.get("job_id") or "").isdigit() else 0
            curr_job = int(row.get("job_id") or 0) if str(row.get("job_id") or "").isdigit() else 0
            if curr_job > prev_job:
                best_by_group[key] = dict(row)

    best_rows = list(best_by_group.values())
    best_rows.sort(
        key=lambda r: (
            float(r.get("best_top1") or -1.0),
            int(r.get("job_id") or 0) if str(r.get("job_id") or "").isdigit() else 0,
        ),
        reverse=True,
    )
    write_csv(path, best_rows)


def main() -> None:
    repo = _repo_root()
    metrics_dir = repo / "mobilefcmvit2" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(repo)

    summary_csv = metrics_dir / "run_metrics_summary.csv"
    success_csv = metrics_dir / "run_metrics_success_only.csv"

    write_csv(summary_csv, rows)
    write_best_csv(success_csv, rows)

    print(f"Wrote {len(rows)} rows to {summary_csv}")
    print(f"Wrote best-only summary to {success_csv}")


if __name__ == "__main__":
    main()
