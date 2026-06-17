from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from poker_agent.event_normalization.schema import (
    Event,
    ExtractionResult,
    build_event,
    model_dump_event,
    parse_event_json,
)


class TrainingExample(BaseModel):
    raw_text: str
    label: Event


def event_from_payload(payload: dict[str, Any]) -> Event:
    return build_event(payload)


def event_to_jsonable(event: Event) -> dict[str, Any]:
    payload = model_dump_event(event)
    payload["cards"] = payload.get("cards") or None
    return payload


def unmatched_event(confidence: float = 0.0) -> Event:
    return build_event(
        {
            "event_type": "unmatched",
            "player": None,
            "action": None,
            "amount": None,
            "cards": None,
            "confidence": confidence,
        }
    )


def validate_model_output(raw_output: str, *, confidence_threshold: float = 0.0) -> ExtractionResult:
    result = parse_event_json(raw_output)
    if result.event.confidence < confidence_threshold:
        result.event = unmatched_event(result.event.confidence)
    return result

