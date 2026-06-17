from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from poker_agent.event_normalization.metrics import amount_equal, compute_metrics, records_to_jsonable
from poker_agent.event_normalization.schema import Event


def card_exact_match(expected: Event, predicted: Event) -> bool:
    return list(expected.cards or []) == list(predicted.cards or [])


def compute_qlora_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_metrics(records)
    if records:
        metrics["card_exact_match"] = float(
            sum(card_exact_match(record["expected"], record["predicted"]) for record in records) / len(records)
        )
        metrics["amount_exact_match_all"] = float(
            sum(amount_equal(record["expected"].amount, record["predicted"].amount) for record in records) / len(records)
        )
    else:
        metrics["card_exact_match"] = 0.0
        metrics["amount_exact_match_all"] = 0.0
    return metrics


def write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records_to_jsonable(records):
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


def write_baseline_comparison(
    *,
    baseline_csv: Path,
    metrics: dict[str, Any],
    out_csv: Path,
    method_name: str = "qwen25_qlora",
) -> None:
    rows: list[dict[str, Any]] = []
    if baseline_csv.exists():
        baseline = pd.read_csv(baseline_csv)
        for row in baseline.to_dict(orient="records"):
            rows.append(
                {
                    "method": row.get("method"),
                    "model": row.get("model"),
                    "accuracy": row.get("accuracy"),
                    "macro_f1": row.get("macro_f1"),
                    "schema_validity_rate": row.get("schema_validity_rate"),
                    "event_type_exact_match": row.get("event_type_exact_match"),
                    "action_exact_match": row.get("action_exact_match"),
                    "amount_exact_match": row.get("amount_exact_match"),
                    "average_latency_ms": row.get("average_latency_ms"),
                    "unmatched_rate": row.get("unmatched_rate"),
                }
            )
    rows.append(
        {
            "method": method_name,
            "model": metrics.get("model", "Qwen/Qwen2.5-1.5B-Instruct"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "schema_validity_rate": metrics.get("schema_validity_rate"),
            "event_type_exact_match": metrics.get("event_type_exact_match"),
            "action_exact_match": metrics.get("action_exact_match"),
            "amount_exact_match": metrics.get("amount_exact_match"),
            "average_latency_ms": metrics.get("average_latency_ms"),
            "unmatched_rate": metrics.get("unmatched_rate"),
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

