from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.event_schema import (
    EVENT_SCHEMA_VERSION,
    event_json_schema,
    normalize_expected_event,
    schema_group_id,
    validate_gold_row,
)


CORRUPTION_TYPES = (
    "value_ocr_noise",
    "amount_text_noise",
    "event_name_noise",
    "card_format_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 1 event schema dataset and grouped splits")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--out", default=Path("evaluation/event_extraction_phase1.jsonl"), type=Path)
    parser.add_argument("--schema-out", default=Path("evaluation/event_schema_v1.json"), type=Path)
    parser.add_argument("--splits-out", default=Path("evaluation/event_extraction_phase1_splits.json"), type=Path)
    parser.add_argument("--report-out", default=Path("reports/event_schema_dataset_report.json"), type=Path)
    parser.add_argument("--markdown-out", default=Path("reports/event_schema_dataset_report.md"), type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corruptions-per-example", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["source_line"] = line_number
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def corrupt_text(text: str) -> str:
    replacements = (
        ("o", "0"),
        ("l", "1"),
        ("i", "1"),
        ("e", "3"),
        ("a", "@"),
    )
    result = text
    for old, new in replacements:
        if old in result.lower():
            index = result.lower().find(old)
            return result[:index] + new + result[index + 1 :]
    return text + "."


def amount_to_noisy_text(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"${amount:.2f}".replace(".", ",")


def apply_corruption(row: dict[str, Any], corruption_type: str) -> dict[str, Any]:
    mutated = copy.deepcopy(row)
    record = mutated.setdefault("record", {})
    payload = record.setdefault("event_value", {})

    if corruption_type == "value_ocr_noise":
        if "value" in payload:
            payload["value"] = corrupt_text(str(payload["value"]))
        elif "player_position" in payload:
            payload["player_position"] = corrupt_text(str(payload["player_position"]))
    elif corruption_type == "amount_text_noise":
        for key in ("amount", "diff", "stack"):
            if key in payload:
                payload[key] = amount_to_noisy_text(payload[key])
                break
    elif corruption_type == "event_name_noise":
        event_name = str(record.get("event_name") or "")
        if event_name:
            record["event_name"] = corrupt_text(event_name)
    elif corruption_type == "card_format_noise":
        cards = payload.get("cards")
        if isinstance(cards, list) and cards:
            payload["cards"] = " , ".join(str(card).lower() for card in cards)
        elif isinstance(cards, str):
            payload["cards"] = cards.lower().replace(" ", " , ")

    return mutated


def materialize_row(
    row: dict[str, Any],
    *,
    row_id: str,
    parent_id: str,
    variant: str,
    noise_type: str,
    noise_severity: int,
) -> dict[str, Any]:
    expected = normalize_expected_event(row["expected"])
    payload = {
        "id": row_id,
        "parent_id": parent_id,
        "group_id": schema_group_id({"id": parent_id, "record": row.get("record", {})}),
        "split": "unassigned",
        "schema_version": EVENT_SCHEMA_VERSION,
        "variant": variant,
        "noise": {
            "type": noise_type,
            "severity": noise_severity,
        },
        "record": row["record"],
        "expected": expected,
    }
    if "source_line" in row:
        payload["source_line"] = row["source_line"]
    return payload


def build_expanded_rows(rows: list[dict[str, Any]], corruptions_per_example: int) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        parent_id = str(row["id"])
        clean = materialize_row(
            row,
            row_id=f"{parent_id}_clean",
            parent_id=parent_id,
            variant="clean",
            noise_type="clean",
            noise_severity=0,
        )
        expanded.append(clean)
        for index in range(max(0, corruptions_per_example)):
            corruption_type = CORRUPTION_TYPES[index % len(CORRUPTION_TYPES)]
            corrupted = apply_corruption(row, corruption_type)
            expanded.append(
                materialize_row(
                    corrupted,
                    row_id=f"{parent_id}_corrupt_{index + 1}",
                    parent_id=parent_id,
                    variant="corrupted",
                    noise_type=corruption_type,
                    noise_severity=1,
                )
            )
    return expanded


def split_groups(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> dict[str, list[str]]:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    labels_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        labels_by_group[str(row["group_id"])][row["expected"]["event_type"]] += 1

    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group_id, labels in labels_by_group.items():
        label = labels.most_common(1)[0][0]
        groups_by_label[label].append(group_id)

    splits = {"train": [], "valid": [], "test": []}
    for label in sorted(groups_by_label):
        groups = sorted(groups_by_label[label])
        rng.shuffle(groups)
        total = len(groups)
        train_count = int(round(total * train_ratio))
        valid_count = int(round(total * valid_ratio))
        if total >= 3:
            train_count = min(max(train_count, 1), total - 2)
            valid_count = min(max(valid_count, 1), total - train_count - 1)
        elif total == 2:
            train_count = 1
            valid_count = 1
        elif total == 1:
            train_count = 1
            valid_count = 0
        test_count = total - train_count - valid_count
        if test_count < 0:
            valid_count = max(0, valid_count + test_count)
            test_count = 0

        splits["train"].extend(groups[:train_count])
        splits["valid"].extend(groups[train_count : train_count + valid_count])
        splits["test"].extend(groups[train_count + valid_count :])

    return {split: sorted(group_ids) for split, group_ids in splits.items()}


def assign_splits(rows: list[dict[str, Any]], splits: dict[str, list[str]]) -> None:
    group_to_split = {
        group_id: split
        for split, group_ids in splits.items()
        for group_id in group_ids
    }
    for row in rows:
        row["split"] = group_to_split.get(str(row["group_id"]), "unassigned")


def validation_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in rows:
        result = validate_gold_row(row)
        if not result.valid:
            errors.append({"id": row.get("id"), "errors": result.errors})
    return errors


def summarize(rows: list[dict[str, Any]], splits: dict[str, list[str]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(row["expected"]["event_type"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    noise_counts = Counter(row["noise"]["type"] for row in rows)
    labels_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        labels_by_split[row["split"]][row["expected"]["event_type"]] += 1
    return {
        "status": "PASS" if not errors else "FAIL",
        "schema_version": EVENT_SCHEMA_VERSION,
        "examples": len(rows),
        "parent_examples": len({row["parent_id"] for row in rows}),
        "groups": len({row["group_id"] for row in rows}),
        "label_counts": dict(sorted(label_counts.items())),
        "noise_counts": dict(sorted(noise_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "labels_by_split": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(labels_by_split.items())
        },
        "splits": {split: len(group_ids) for split, group_ids in sorted(splits.items())},
        "validation_errors": errors,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = [
        "# Phase 1 Event Schema Dataset",
        "",
        "## Summary",
        "",
        f"- Status: `{report['status']}`",
        f"- Schema version: `{report['schema_version']}`",
        f"- Parent examples: `{report['parent_examples']}`",
        f"- Expanded examples: `{report['examples']}`",
        f"- Groups: `{report['groups']}`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in report["label_counts"].items():
        rows.append(f"| {label} | {count} |")
    rows.extend(["", "## Split Counts", "", "| Split | Rows |", "| --- | ---: |"])
    for split, count in report["split_counts"].items():
        rows.append(f"| {split} | {count} |")
    rows.extend(["", "## Noise Counts", "", "| Noise | Count |", "| --- | ---: |"])
    for noise, count in report["noise_counts"].items():
        rows.append(f"| {noise} | {count} |")
    rows.extend(
        [
            "",
            "## Validation",
            "",
            "All rows are validated against the internal event schema before writing the dataset."
            if report["status"] == "PASS"
            else "Validation errors were found; see the JSON report.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    clean_rows = read_jsonl(args.gold)
    source_errors = validation_errors(clean_rows)
    if source_errors:
        raise SystemExit(json.dumps({"status": "FAIL", "source_validation_errors": source_errors}, indent=2))

    expanded_rows = build_expanded_rows(clean_rows, corruptions_per_example=args.corruptions_per_example)
    splits = split_groups(
        expanded_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )
    assign_splits(expanded_rows, splits)
    errors = validation_errors(expanded_rows)
    report = summarize(expanded_rows, splits, errors)

    args.schema_out.parent.mkdir(parents=True, exist_ok=True)
    args.schema_out.write_text(json.dumps(event_json_schema(), indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(expanded_rows, args.out)
    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    args.splits_out.write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.markdown_out)

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
