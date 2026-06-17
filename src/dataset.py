from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from poker_agent.event_normalization.benchmark import expected_event_from_row, record_to_raw_text

from src.schema import TrainingExample, event_to_jsonable


SPLIT_MAP = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def phase1_row_to_training_example(row: dict[str, Any]) -> dict[str, Any]:
    raw_text = record_to_raw_text(row)
    label = expected_event_from_row(row)
    example = TrainingExample(raw_text=raw_text, label=label)
    return {
        "id": row.get("id") or f"line_{row.get('_line_number')}",
        "group_id": row.get("group_id") or row.get("parent_id") or row.get("id"),
        "raw_text": example.raw_text,
        "label": event_to_jsonable(example.label),
        "source_split": row.get("split"),
        "schema_version": row.get("schema_version"),
        "noise": row.get("noise") or {},
    }


def build_qlora_splits(
    *,
    source: Path,
    output_dir: Path,
    seed: int = 42,
    shuffle_train: bool = True,
) -> dict[str, Any]:
    raw_rows = read_jsonl(source)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in raw_rows:
        split = SPLIT_MAP.get(str(row.get("split") or "").lower())
        if split is None:
            continue
        buckets[split].append(phase1_row_to_training_example(row))

    if shuffle_train:
        random.Random(seed).shuffle(buckets["train"])

    counts = {
        split: write_jsonl(output_dir / f"{split}.jsonl", rows)
        for split, rows in buckets.items()
    }
    event_type_counts = {
        split: dict(Counter(row["label"]["event_type"] for row in rows))
        for split, rows in buckets.items()
    }
    manifest = {
        "source": str(source),
        "output_dir": str(output_dir),
        "seed": seed,
        "files": {split: str((output_dir / f"{split}.jsonl").as_posix()) for split in buckets},
        "counts": counts,
        "event_type_counts": event_type_counts,
        "status": "PASS" if all(counts.values()) else "FAIL",
    }
    manifest_path = output_dir / "qlora_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_supervised_rows(path: Path, *, max_examples: int = 0) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_examples > 0:
        return rows[:max_examples]
    return rows


def load_hf_json_dataset(train_file: Path, val_file: Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for full training.") from exc

    return load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )

