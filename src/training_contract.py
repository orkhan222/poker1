from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.dataset import load_supervised_rows
from src.schema import event_from_payload


REQUIRED_ROW_KEYS = ("raw_text", "label")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_group_id(row: dict[str, Any]) -> str:
    return str(row.get("group_id") or row.get("id") or row.get("raw_text") or "").strip()


def row_text_key(row: dict[str, Any]) -> str:
    return str(row.get("raw_text") or "").strip().lower()


def validate_supervised_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_ROW_KEYS:
        if key not in row:
            errors.append(f"missing {key}")
    raw_text = str(row.get("raw_text") or "").strip()
    if not raw_text:
        errors.append("raw_text is empty")
    label = row.get("label")
    if not isinstance(label, dict):
        errors.append("label must be an object")
        return errors
    try:
        event = event_from_payload(label)
    except Exception as exc:
        errors.append(f"label cannot be normalized: {exc}")
        return errors
    if event.event_type == "player_action" and event.action is None:
        errors.append("player_action requires action")
    if event.event_type == "card_event" and not event.normalized_cards:
        errors.append("card_event requires cards")
    if event.event_type in {"pot_event", "dealer_event"} and event.amount is None:
        errors.append(f"{event.event_type} requires amount")
    if event.event_type == "unmatched" and any([event.player, event.action, event.amount, event.normalized_cards]):
        errors.append("unmatched must not carry semantic fields")
    return errors


def profile_supervised_split(path: Path, *, max_examples: int = 0) -> dict[str, Any]:
    rows = load_supervised_rows(path, max_examples=max_examples)
    invalid_rows: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    raw_text_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        errors = validate_supervised_row(row)
        if errors:
            invalid_rows.append({"index": index, "id": row.get("id"), "errors": errors})
            continue
        event = event_from_payload(row["label"])
        event_counts[event.event_type] += 1
        if event.action:
            action_counts[event.action] += 1
        confidence_values.append(float(event.confidence))
        text_key = row_text_key(row)
        group_id = row_group_id(row)
        if text_key:
            raw_text_counts[text_key] += 1
        if group_id:
            group_counts[group_id] += 1

    duplicate_texts = [key for key, count in raw_text_counts.items() if count > 1]
    duplicate_groups = [key for key, count in group_counts.items() if count > 1]
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "rows": len(rows),
        "valid_rows": len(rows) - len(invalid_rows),
        "invalid_rows": invalid_rows[:50],
        "invalid_row_count": len(invalid_rows),
        "schema_validity_rate": (len(rows) - len(invalid_rows)) / len(rows) if rows else 0.0,
        "event_type_counts": dict(event_counts),
        "action_counts": dict(action_counts),
        "mean_label_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "duplicate_raw_text_count": len(duplicate_texts),
        "duplicate_group_count": len(duplicate_groups),
        "raw_text_keys": sorted(raw_text_counts),
        "group_ids": sorted(group_counts),
    }


def leakage_report(split_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    split_names = sorted(split_profiles)
    overlaps: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            left = split_profiles[left_name]
            right = split_profiles[right_name]
            text_overlap = sorted(set(left["raw_text_keys"]) & set(right["raw_text_keys"]))
            group_overlap = sorted(set(left["group_ids"]) & set(right["group_ids"]))
            if text_overlap or group_overlap:
                overlaps.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "raw_text_overlap_count": len(text_overlap),
                        "group_overlap_count": len(group_overlap),
                        "raw_text_overlap_sample": text_overlap[:20],
                        "group_overlap_sample": group_overlap[:20],
                    }
                )
    return {"overlap_count": len(overlaps), "overlaps": overlaps}


def class_balance_report(split_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split, profile in split_profiles.items():
        counts = profile["event_type_counts"]
        total = sum(counts.values())
        if total <= 0:
            payload[split] = {"status": "FAIL", "dominant_event_type": None, "dominant_ratio": 0.0}
            continue
        dominant_label, dominant_count = max(counts.items(), key=lambda item: item[1])
        payload[split] = {
            "status": "PASS",
            "dominant_event_type": dominant_label,
            "dominant_ratio": dominant_count / total,
            "event_type_distribution": {key: value / total for key, value in sorted(counts.items())},
        }
    return payload


def dataset_contract_status(
    *,
    profiles: dict[str, dict[str, Any]],
    leakage: dict[str, Any],
    min_rows: dict[str, int],
    min_schema_validity: float,
    max_dominant_ratio: float,
) -> tuple[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    balance = class_balance_report(profiles)
    for split, profile in profiles.items():
        if profile["rows"] < int(min_rows.get(split, 1)):
            errors.append(f"{split} split has {profile['rows']} rows; expected at least {min_rows.get(split, 1)}")
        if profile["schema_validity_rate"] < min_schema_validity:
            errors.append(
                f"{split} schema validity {profile['schema_validity_rate']:.4f} is below {min_schema_validity:.4f}"
            )
        dominant_ratio = balance.get(split, {}).get("dominant_ratio", 0.0)
        if dominant_ratio > max_dominant_ratio:
            warnings.append(f"{split} dominant class ratio {dominant_ratio:.4f} exceeds {max_dominant_ratio:.4f}")
    if leakage["overlap_count"]:
        errors.append("train/validation/test leakage detected")
    if errors:
        return "FAIL", errors + warnings
    if warnings:
        return "WARN", warnings
    return "PASS", []


def validate_training_contract(
    *,
    train_file: Path,
    val_file: Path,
    test_file: Path,
    min_rows: dict[str, int] | None = None,
    min_schema_validity: float = 1.0,
    max_dominant_ratio: float = 0.75,
    max_examples: int = 0,
) -> dict[str, Any]:
    paths = {"train": train_file, "val": val_file, "test": test_file}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {"status": "FAIL", "errors": [f"missing file: {path}" for path in missing], "splits": {}}
    profiles = {
        split: profile_supervised_split(path, max_examples=max_examples)
        for split, path in paths.items()
    }
    for profile in profiles.values():
        profile.pop("raw_text_keys", None)
        profile.pop("group_ids", None)
    raw_profiles = {
        split: profile_supervised_split(path, max_examples=max_examples)
        for split, path in paths.items()
    }
    leakage = leakage_report(raw_profiles)
    balance = class_balance_report(raw_profiles)
    status, messages = dataset_contract_status(
        profiles=raw_profiles,
        leakage=leakage,
        min_rows=min_rows or {"train": 1, "val": 1, "test": 1},
        min_schema_validity=min_schema_validity,
        max_dominant_ratio=max_dominant_ratio,
    )
    return {
        "status": status,
        "messages": messages,
        "splits": profiles,
        "leakage": leakage,
        "class_balance": balance,
    }


def write_contract_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
