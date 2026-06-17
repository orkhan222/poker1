from __future__ import annotations

import json
from typing import Any

from src.prompts import build_prompt
from src.schema import Event, event_to_jsonable


def completion_text(label: Event) -> str:
    return json.dumps(event_to_jsonable(label), ensure_ascii=False, sort_keys=True)


def training_text(raw_text: str, label: Event, *, eos_token: str = "") -> str:
    return build_prompt(raw_text) + completion_text(label) + eos_token


def format_training_row(row: dict[str, Any], *, eos_token: str = "") -> dict[str, str]:
    label = row["label"]
    if not isinstance(label, Event):
        from src.schema import event_from_payload

        label = event_from_payload(label)
    return {"text": training_text(str(row["raw_text"]), label, eos_token=eos_token)}

