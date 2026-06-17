from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


EventType = Literal["player_action", "card_event", "pot_event", "dealer_event", "unmatched"]
Action = Literal["fold", "check", "call", "bet", "raise", "all_in", "small_blind", "big_blind"]

EVENT_TYPES: tuple[str, ...] = ("player_action", "card_event", "pot_event", "dealer_event", "unmatched")
ACTIONS: tuple[str, ...] = ("fold", "check", "call", "bet", "raise", "all_in", "small_blind", "big_blind")
CARD_RE = re.compile(r"^[2-9TJQKA][CDHS]$")

EVENT_TYPE_ALIASES = {
    "card_update": "card_event",
    "card": "card_event",
    "cards": "card_event",
    "pot_update": "pot_event",
    "stack_update": "pot_event",
    "stack": "pot_event",
    "dealer_message": "dealer_event",
    "dealer": "dealer_event",
}

ACTION_ALIASES = {
    "all-in": "all_in",
    "all in": "all_in",
    "allin": "all_in",
    "small blind": "small_blind",
    "small_blind": "small_blind",
    "sb": "small_blind",
    "big blind": "big_blind",
    "big_blind": "big_blind",
    "bb": "big_blind",
    "raises": "raise",
    "raised": "raise",
    "bets": "bet",
    "calls": "call",
    "checks": "check",
    "folds": "fold",
}

SUIT_ALIASES = {
    "♠": "S",
    "♤": "S",
    "spades": "S",
    "spade": "S",
    "s": "S",
    "♥": "H",
    "♡": "H",
    "hearts": "H",
    "heart": "H",
    "h": "H",
    "♦": "D",
    "♢": "D",
    "diamonds": "D",
    "diamond": "D",
    "d": "D",
    "♣": "C",
    "♧": "C",
    "clubs": "C",
    "club": "C",
    "c": "C",
}


class Event(BaseModel):
    event_type: EventType = "unmatched"
    player: str | None = None
    action: Action | None = None
    amount: float | None = None
    cards: list[str] | None = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def normalized_cards(self) -> list[str]:
        return list(self.cards or [])


@dataclass
class ExtractionResult:
    event: Event
    raw_output: str = ""
    schema_valid: bool = True
    errors: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    candidate_count: int = 0


def model_dump_event(event: Event) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    return event.dict()


def normalize_event_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = EVENT_TYPE_ALIASES.get(text, text)
    return text if text in EVENT_TYPES else "unmatched"


def normalize_action(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = ACTION_ALIASES.get(text, text)
    text = text.replace(" ", "_")
    return text if text in ACTIONS else None


def normalize_amount(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        amount = float(str(value).replace("$", "").replace(",", ".").strip())
    except ValueError:
        return None
    return amount


def normalize_player(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    match = re.search(r"(?:player|plyr|p1ayer)\s*[_-]?\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"Player{match.group(1)}"
    return text


def normalize_card(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    for source, target in SUIT_ALIASES.items():
        text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
    text = text.upper().replace("10", "T")
    text = re.sub(r"[^2-9TJQKACDHS]", "", text)
    if len(text) >= 2:
        candidate = text[0] + text[-1]
        if CARD_RE.match(candidate):
            return candidate
    return None


def normalize_cards(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[\s,;/]+", value)
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]
    cards: list[str] = []
    for raw in raw_items:
        card = normalize_card(raw)
        if card and card not in cards:
            cards.append(card)
    return cards


def build_event(payload: dict[str, Any]) -> Event:
    event_type = normalize_event_type(payload.get("event_type"))
    action = normalize_action(payload.get("action"))
    amount = normalize_amount(payload.get("amount"))
    cards = normalize_cards(payload.get("cards"))
    player = normalize_player(payload.get("player"))

    if event_type != "player_action":
        action = None
    if event_type != "card_event":
        cards = []
    if event_type not in {"player_action", "pot_event", "dealer_event"}:
        amount = None
    if event_type == "unmatched":
        player = None
        action = None
        amount = None
        cards = []

    confidence = payload.get("confidence", 0.0)
    try:
        confidence_float = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_float = 0.0

    return Event(
        event_type=event_type,
        player=player,
        action=action,
        amount=amount,
        cards=cards,
        confidence=confidence_float,
    )


def parse_event_json(raw_output: str) -> ExtractionResult:
    raw_text = str(raw_output or "").strip()
    if raw_text and not raw_text.startswith("{"):
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw_text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(raw_text[index:])
                raw_output = json.dumps(payload)
                break
            except json.JSONDecodeError:
                continue
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return ExtractionResult(
            event=Event(event_type="unmatched", confidence=0.0),
            raw_output=raw_output,
            schema_valid=False,
            errors=[str(exc)],
        )
    if not isinstance(payload, dict):
        return ExtractionResult(
            event=Event(event_type="unmatched", confidence=0.0),
            raw_output=raw_output,
            schema_valid=False,
            errors=["model output is not a JSON object"],
        )
    try:
        event = build_event(payload)
    except (TypeError, ValidationError, ValueError) as exc:
        return ExtractionResult(
            event=Event(event_type="unmatched", confidence=0.0),
            raw_output=raw_output,
            schema_valid=False,
            errors=[str(exc)],
        )
    return ExtractionResult(event=event, raw_output=raw_output, schema_valid=True)


def event_key(event: Event, *, amount_tolerance: float = 1e-6) -> tuple[Any, ...]:
    amount = event.amount
    rounded = None if amount is None else round(float(amount) / amount_tolerance) * amount_tolerance
    return (
        event.event_type,
        normalize_player(event.player),
        normalize_action(event.action),
        None if rounded is None else round(rounded, 6),
        tuple(event.normalized_cards),
    )
