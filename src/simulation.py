from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from poker_agent.event_normalization.schema import Event, build_event, model_dump_event


SIMULATION_SCHEMA_VERSION = "simulation_event_v1"
SIMULATION_EVENT_TYPES = {"player_action", "card_event", "pot_event", "dealer_event"}


def event_from_any(value: Any) -> Event:
    if isinstance(value, Event):
        return value
    if isinstance(value, dict):
        return build_event(value)
    raise TypeError(f"Unsupported event payload: {type(value)!r}")


def simulation_rejection_reason(
    event: Event,
    *,
    schema_valid: bool,
    min_confidence: float,
) -> str | None:
    if not schema_valid:
        return "schema_invalid"
    if event.event_type == "unmatched":
        return "unmatched"
    if event.event_type not in SIMULATION_EVENT_TYPES:
        return "unsupported_event_type"
    if float(event.confidence) < min_confidence:
        return "low_confidence"
    if event.event_type == "player_action" and event.action is None:
        return "missing_action"
    if event.event_type == "card_event" and not event.normalized_cards:
        return "missing_cards"
    if event.event_type in {"pot_event", "dealer_event"} and event.amount is None:
        return "missing_amount"
    return None


def simulation_event_from_record(
    record: dict[str, Any],
    *,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    event = event_from_any(record["predicted"])
    schema_valid = bool(record.get("schema_valid", True))
    rejection = simulation_rejection_reason(event, schema_valid=schema_valid, min_confidence=min_confidence)
    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "id": record.get("id"),
        "raw_text": record.get("raw_text"),
        "event": model_dump_event(event),
        "simulation_ready": rejection is None,
        "rejection_reason": rejection,
        "source": {
            "method": record.get("method"),
            "model": record.get("model"),
            "latency_ms": record.get("latency_ms"),
            "errors": record.get("errors") or [],
        },
    }


def simulation_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [event for event in events if event["simulation_ready"]]
    rejected = [event for event in events if not event["simulation_ready"]]
    event_counts = Counter(event["event"]["event_type"] for event in ready)
    rejection_counts = Counter(event["rejection_reason"] for event in rejected)
    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "total_events": len(events),
        "ready_events": len(ready),
        "rejected_events": len(rejected),
        "readiness_rate": len(ready) / len(events) if events else 0.0,
        "ready_event_type_counts": dict(event_counts),
        "rejection_counts": dict(rejection_counts),
    }


def export_simulation_events(
    *,
    records: list[dict[str, Any]],
    out_jsonl: Path,
    summary_out: Path,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    events = [
        simulation_event_from_record(record, min_confidence=min_confidence)
        for record in records
    ]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    summary = simulation_summary(events)
    summary["output_file"] = str(out_jsonl)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "predicted" in payload:
                payload["predicted"] = build_event(payload["predicted"])
            if "expected" in payload:
                payload["expected"] = build_event(payload["expected"])
            records.append(payload)
    return records
