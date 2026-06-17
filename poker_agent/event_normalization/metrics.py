from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from poker_agent.event_normalization.schema import EVENT_TYPES, Event, event_key, model_dump_event


def amount_equal(left: float | None, right: float | None, tolerance: float = 1e-6) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) <= tolerance


def exact_event_match(expected: Event, predicted: Event) -> bool:
    return event_key(expected) == event_key(predicted)


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "examples": 0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "macro_f1": 0.0,
            "schema_validity_rate": 0.0,
            "action_exact_match": 0.0,
            "amount_exact_match": 0.0,
            "event_type_exact_match": 0.0,
            "average_latency_ms": 0.0,
            "peak_memory_mb": 0.0,
            "unmatched_rate": 0.0,
        }

    expected_events = [record["expected"] for record in records]
    predicted_events = [record["predicted"] for record in records]
    y_true = [event.event_type for event in expected_events]
    y_pred = [event.event_type for event in predicted_events]
    labels = [label for label in EVENT_TYPES if label in set(y_true) | set(y_pred)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    exact_matches = [exact_event_match(expected, predicted) for expected, predicted in zip(expected_events, predicted_events)]
    action_support = [index for index, event in enumerate(expected_events) if event.action is not None]
    amount_support = [index for index, event in enumerate(expected_events) if event.amount is not None]

    return {
        "examples": len(records),
        "accuracy": float(np.mean(exact_matches)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "schema_validity_rate": float(np.mean([record["schema_valid"] for record in records])),
        "action_exact_match": float(
            np.mean([expected_events[index].action == predicted_events[index].action for index in action_support])
        )
        if action_support
        else 0.0,
        "amount_exact_match": float(
            np.mean([amount_equal(expected_events[index].amount, predicted_events[index].amount) for index in amount_support])
        )
        if amount_support
        else 0.0,
        "event_type_exact_match": float(np.mean([left == right for left, right in zip(y_true, y_pred)])),
        "average_latency_ms": float(np.mean([record["latency_ms"] for record in records])),
        "peak_memory_mb": float(max(record["peak_memory_mb"] for record in records)),
        "unmatched_rate": float(np.mean([event.event_type == "unmatched" for event in predicted_events])),
        "event_type_counts": dict(Counter(y_true)),
        "predicted_event_type_counts": dict(Counter(y_pred)),
        "per_event_type": {
            label: {
                "precision": float(precision[labels.index(label)]) if label in labels else 0.0,
                "recall": float(recall[labels.index(label)]) if label in labels else 0.0,
                "f1": float(f1[labels.index(label)]) if label in labels else 0.0,
                "support": int(support[labels.index(label)]) if label in labels else 0,
            }
            for label in EVENT_TYPES
        },
    }


def records_to_jsonable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **{key: value for key, value in record.items() if key not in {"expected", "predicted"}},
            "expected": model_dump_event(record["expected"]),
            "predicted": model_dump_event(record["predicted"]),
        }
        for record in records
    ]
