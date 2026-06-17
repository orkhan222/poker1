from __future__ import annotations

import argparse
import json
import random
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from poker_agent.event_normalization.backends import create_backend
from poker_agent.event_normalization.candidate_ranker import CandidateRanker
from poker_agent.event_normalization.few_shot import FewShotExtractor
from poker_agent.event_normalization.metrics import compute_metrics, records_to_jsonable
from poker_agent.event_normalization.parser import DeterministicParser
from poker_agent.event_normalization.schema import Event, ExtractionResult, build_event, model_dump_event
from poker_agent.event_normalization.zero_shot import ZeroShotExtractor


MODEL_REGISTRY = {
    "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "smollm2_1_7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "heuristic_text": "heuristic_text",
}


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_model_ids(value: str) -> list[str]:
    resolved: list[str] = []
    for item in split_csv(value):
        resolved.append(MODEL_REGISTRY.get(item, item))
    return resolved


def record_to_raw_text(row: dict[str, Any]) -> str:
    if str(row.get("raw_text") or "").strip():
        return str(row["raw_text"]).strip()
    record = row.get("record") if isinstance(row.get("record"), dict) else {}
    payload = record.get("event_value") if isinstance(record.get("event_value"), dict) else {}
    parts: list[str] = []
    for value in (record.get("event_name"), record.get("object_type")):
        if value not in (None, ""):
            parts.append(str(value))
    for key in ("player_position", "player", "nickname", "value", "action", "cards", "amount", "stack", "diff", "pot"):
        value = payload.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            parts.append(" ".join(str(item) for item in value))
        else:
            parts.append(str(value))
    return " ".join(parts).strip()


def expected_event_from_row(row: dict[str, Any]) -> Event:
    expected = dict(row.get("expected") or {})
    record = row.get("record") if isinstance(row.get("record"), dict) else {}
    payload = record.get("event_value") if isinstance(record.get("event_value"), dict) else {}
    raw_value = str(payload.get("value") or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw_value = raw_value.replace("p0st", "post")
    if raw_value in {"post_sb", "posts_sb", "sb"}:
        expected.update({"event_type": "player_action", "action": "small_blind", "amount": payload.get("amount")})
    elif raw_value in {"post_bb", "posts_bb", "bb"}:
        expected.update({"event_type": "player_action", "action": "big_blind", "amount": payload.get("amount")})
    elif raw_value in {"all_in", "allin"}:
        expected.update({"event_type": "player_action", "action": "all_in", "amount": payload.get("amount")})
    expected.setdefault("player", payload.get("player_position") or payload.get("player"))
    expected.setdefault("confidence", 1.0)
    return build_event(expected)


def load_dataset(path: Path, split: str, max_examples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and row.get("split") != split:
                continue
            row["line_number"] = line_number
            row["raw_text"] = record_to_raw_text(row)
            row["expected_event"] = expected_event_from_row(row)
            rows.append(row)
    return rows[:max_examples] if max_examples > 0 else rows


def build_few_shot_examples(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["expected_event"].event_type, []).append(row)
    rng = random.Random(seed)
    for bucket in grouped.values():
        rng.shuffle(bucket)
    examples: list[dict[str, Any]] = []
    while len(examples) < size and any(grouped.values()):
        for label in sorted(grouped):
            if grouped[label] and len(examples) < size:
                row = grouped[label].pop()
                examples.append({"raw_text": row["raw_text"], "expected": row["expected_event"]})
    return examples


def run_system(
    *,
    method: str,
    model_id: str,
    rows: list[dict[str, Any]],
    extractor: Callable[[str], ExtractionResult],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    tracemalloc.start()
    for row in rows:
        tracemalloc.reset_peak()
        started = time.perf_counter()
        result = extractor(row["raw_text"])
        latency_ms = (time.perf_counter() - started) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        result.latency_ms = latency_ms
        records.append(
            {
                "id": row.get("id"),
                "method": method,
                "model": model_id,
                "raw_text": row["raw_text"],
                "expected": row["expected_event"],
                "predicted": result.event,
                "schema_valid": result.schema_valid,
                "errors": result.errors,
                "raw_output": result.raw_output,
                "latency_ms": latency_ms,
                "peak_memory_mb": peak_bytes / (1024 * 1024),
                "candidate_count": result.candidate_count,
            }
        )
    tracemalloc.stop()
    metrics = compute_metrics(records)
    metrics.update({"method": method, "model": model_id})
    return metrics, records


def parser_extractor(parser: DeterministicParser) -> Callable[[str], ExtractionResult]:
    def extract(text: str) -> ExtractionResult:
        return ExtractionResult(event=parser.parse(text), raw_output="", schema_valid=True)

    return extract


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    all_train_rows = load_dataset(args.dataset, "train", 0)
    eval_rows = load_dataset(args.dataset, args.split, args.max_examples)
    if not eval_rows:
        raise ValueError(f"No rows found for split={args.split} in {args.dataset}")

    few_shot_sizes = [int(item) for item in split_csv(args.few_shot_sizes)]
    model_ids = resolve_model_ids(args.model_ids)
    deterministic_parser = DeterministicParser(ocr_profile="full")

    systems: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    metrics, records = run_system(
        method="deterministic_parser",
        model_id="-",
        rows=eval_rows,
        extractor=parser_extractor(deterministic_parser),
    )
    systems.append(metrics)
    prediction_records.extend(records)

    for model_id in model_ids:
        zero_backend = create_backend(
            backend=args.backend,
            model_id=model_id,
            parser_profile="light",
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        zero = ZeroShotExtractor(zero_backend)
        metrics, records = run_system(
            method="zero_shot",
            model_id=model_id,
            rows=eval_rows,
            extractor=zero.extract,
        )
        systems.append(metrics)
        prediction_records.extend(records)

        for size in few_shot_sizes:
            examples = build_few_shot_examples(all_train_rows, size, args.seed)
            few_backend = create_backend(
                backend=args.backend,
                model_id=model_id,
                parser_profile="full",
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            few = FewShotExtractor(few_backend, examples)
            metrics, records = run_system(
                method=f"few_shot_{size}",
                model_id=model_id,
                rows=eval_rows,
                extractor=few.extract,
            )
            systems.append(metrics)
            prediction_records.extend(records)

        rank_backend = create_backend(
            backend=args.backend,
            model_id=model_id,
            parser_profile="full",
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        ranker = CandidateRanker(rank_backend, deterministic_parser)
        metrics, records = run_system(
            method="candidate_ranker",
            model_id=model_id,
            rows=eval_rows,
            extractor=ranker.extract,
        )
        systems.append(metrics)
        prediction_records.extend(records)

    summary = pd.DataFrame(systems).sort_values(["macro_f1", "accuracy"], ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.out_csv, index=False)
    json_payload = {
        "dataset": str(args.dataset),
        "split": args.split,
        "backend": args.backend,
        "model_ids": model_ids,
        "examples": len(eval_rows),
        "seed": args.seed,
        "systems": systems,
    }
    args.out_json.write_text(json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8")

    with args.predictions_out.open("w", encoding="utf-8") as handle:
        for record in records_to_jsonable(prediction_records):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    args.report_out.write_text(build_report(json_payload, summary), encoding="utf-8")
    return json_payload


def build_report(payload: dict[str, Any], summary: pd.DataFrame) -> str:
    columns = [
        "method",
        "model",
        "accuracy",
        "macro_f1",
        "precision_macro",
        "recall_macro",
        "schema_validity_rate",
        "event_type_exact_match",
        "action_exact_match",
        "amount_exact_match",
        "average_latency_ms",
        "peak_memory_mb",
        "unmatched_rate",
    ]
    table_rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary[columns].to_dict(orient="records"):
        formatted = []
        for column in columns:
            value = row[column]
            formatted.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table_rows.append("| " + " | ".join(formatted) + " |")
    table = "\n".join(table_rows)
    best = summary.iloc[0].to_dict()
    return "\n".join(
        [
            "# Phase 2 Event-Normalization Baseline Benchmark",
            "",
            "## Objective",
            "",
            "Compare deterministic parsing, zero-shot extraction, few-shot extraction, and candidate ranking on the same grouped OCR/dealer-log evaluation split.",
            "",
            "## Dataset",
            "",
            f"- Dataset: `{payload['dataset']}`",
            f"- Split: `{payload['split']}`",
            f"- Examples: `{payload['examples']}`",
            f"- Backend: `{payload['backend']}`",
            f"- Models: `{', '.join(payload['model_ids'])}`",
            "",
            "## Results",
            "",
            table,
            "",
            "## Best System",
            "",
            f"`{best['method']}` with model `{best['model']}` reached macro F1 `{best['macro_f1']:.4f}` and exact-event accuracy `{best['accuracy']:.4f}`.",
            "",
            "## Notes",
            "",
            "The default backend is deterministic and intended for reproducible local delivery checks. Use `--backend transformers` with the configured model IDs to run the same benchmark against Qwen or SmolLM models when the model weights are available locally or can be downloaded.",
            "",
        ]
    )


def main(args: argparse.Namespace) -> None:
    payload = benchmark(args)
    best = max(payload["systems"], key=lambda item: item["macro_f1"])
    print(f"phase2_examples={payload['examples']}")
    print(f"phase2_best_method={best['method']}")
    print(f"phase2_best_model={best['model']}")
    print(f"phase2_best_macro_f1={best['macro_f1']:.6f}")
    print(f"phase2_best_accuracy={best['accuracy']:.6f}")
    print(json.dumps(payload, indent=2, sort_keys=True))
