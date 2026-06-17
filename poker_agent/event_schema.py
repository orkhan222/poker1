from __future__ import annotations

import re
from dataclasses import dataclass
<<<<<<< HEAD
from typing import Any
=======
<<<<<<< HEAD
from typing import Any, Iterable
>>>>>>> 722cfd2 (LLM training validation simulation)

from poker_agent.features import normalize_action, parse_cards, safe_float, safe_int


EVENT_SCHEMA_VERSION = "event_schema_v1"
EVENT_TYPES = ("player_action", "card_update", "stack_update", "unmatched")
DECISION_ACTIONS = ("fold", "call", "check", "bet", "raise")
CARD_PATTERN = re.compile(r"^[2-9TJQKA][cdhsCDHS]$")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def normalize_event_type(raw: Any) -> str:
    event_type = str(raw or "").strip().lower()
    return event_type if event_type in EVENT_TYPES else "unmatched"


<<<<<<< HEAD
def normalize_event_action(raw: Any) -> str | None:
    if raw in (None, ""):
=======
def issue(code: str, field: str, message: str) -> SchemaIssue:
    return SchemaIssue(code=code, field=field, message=message)


def canonical_event_action(raw: Any) -> str | None:
    if raw is None:
=======
from typing import Any

from poker_agent.features import normalize_action, parse_cards, safe_float, safe_int


EVENT_SCHEMA_VERSION = "event_schema_v1"
EVENT_TYPES = ("player_action", "card_update", "stack_update", "unmatched")
DECISION_ACTIONS = ("fold", "call", "check", "bet", "raise")
CARD_PATTERN = re.compile(r"^[2-9TJQKA][cdhsCDHS]$")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def normalize_event_type(raw: Any) -> str:
    event_type = str(raw or "").strip().lower()
    return event_type if event_type in EVENT_TYPES else "unmatched"


def normalize_event_action(raw: Any) -> str | None:
    if raw in (None, ""):
>>>>>>> 249bf7f (LLM training/validation/simulation)
>>>>>>> 722cfd2 (LLM training validation simulation)
        return None
    action = normalize_action(str(raw))
    if action == "all_in":
        return "raise"
<<<<<<< HEAD
    return action if action in DECISION_ACTIONS else None
=======
<<<<<<< HEAD
    if action in DECISION_ACTIONS or action in NON_DECISION_ACTIONS:
        return action
    return action or None
>>>>>>> 722cfd2 (LLM training validation simulation)


def normalize_event_amount(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    amount = safe_float(raw, default=0.0)
    return amount if amount != 0.0 else None


def normalize_event_cards(raw: Any) -> list[str]:
    normalized: list[str] = []
    for card in parse_cards(raw):
        text = str(card).strip()
        if len(text) >= 2:
            normalized.append(text[0].upper() + text[1].lower())
    return normalized


def normalize_expected_event(raw: dict[str, Any]) -> dict[str, Any]:
    event_type = normalize_event_type(raw.get("event_type"))
    action = normalize_event_action(raw.get("action"))
    cards = normalize_event_cards(raw.get("cards"))
    amount = normalize_event_amount(raw.get("amount"))

    if event_type != "player_action":
        action = None
    if event_type != "card_update":
        cards = []
    if event_type not in {"player_action", "stack_update"}:
        amount = None

    return {
        "event_type": event_type,
        "action": action,
        "cards": cards,
        "amount": amount,
    }


def validate_expected_event(expected: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    normalized = normalize_expected_event(expected)
    event_type = normalized["event_type"]

    if event_type not in EVENT_TYPES:
        errors.append(f"unsupported event_type={event_type}")
    if event_type == "player_action" and normalized["action"] not in DECISION_ACTIONS:
        errors.append("player_action requires action in the decision label space")
    if event_type != "player_action" and normalized["action"] is not None:
        errors.append("non-action event must not carry an action label")
    if event_type == "card_update":
        if not normalized["cards"]:
            errors.append("card_update requires at least one card")
        invalid_cards = [card for card in normalized["cards"] if not CARD_PATTERN.match(card)]
        if invalid_cards:
            errors.append(f"invalid cards={invalid_cards}")
    if event_type != "card_update" and normalized["cards"]:
        errors.append("non-card event must not carry card labels")
    if event_type == "stack_update" and normalized["amount"] is None:
        errors.append("stack_update requires amount")
    if event_type == "unmatched":
        if normalized["action"] is not None or normalized["cards"] or normalized["amount"] is not None:
            errors.append("unmatched event must not carry action, cards, or amount")
    return ValidationResult(valid=not errors, errors=errors)


<<<<<<< HEAD
=======
def source_consistency_errors(
    source_record: dict[str, Any],
    event_type: str,
    cards: list[str],
) -> list[SchemaIssue]:
    errors: list[SchemaIssue] = []
    event_name = str(source_record.get("event_name") or "")
    object_type = str(source_record.get("object_type") or "")
    if event_name == "ocr_action" and event_type not in {"player_action", "unmatched"}:
        errors.append(issue("source_event_type_mismatch", "event_type", "ocr_action can only produce player_action or unmatched."))
    if event_name == "recognize_cards" and event_type not in {"card_update", "unmatched"}:
        errors.append(issue("source_event_type_mismatch", "event_type", "recognize_cards can only produce card_update or unmatched."))
    if event_name == "ocr_stack" and event_type not in {"stack_update", "unmatched"}:
        errors.append(issue("source_event_type_mismatch", "event_type", "ocr_stack can only produce stack_update or unmatched."))
    if event_type == "card_update" and object_type == "player" and len(cards) > 2:
        errors.append(issue("too_many_player_cards", "cards", "Player card update cannot contain more than two cards."))
    if event_type == "card_update" and object_type == "table" and len(cards) > 5:
        errors.append(issue("too_many_board_cards", "cards", "Table card update cannot contain more than five cards."))
    return errors
=======
    return action if action in DECISION_ACTIONS else None


def normalize_event_amount(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    amount = safe_float(raw, default=0.0)
    return amount if amount != 0.0 else None


def normalize_event_cards(raw: Any) -> list[str]:
    normalized: list[str] = []
    for card in parse_cards(raw):
        text = str(card).strip()
        if len(text) >= 2:
            normalized.append(text[0].upper() + text[1].lower())
    return normalized


def normalize_expected_event(raw: dict[str, Any]) -> dict[str, Any]:
    event_type = normalize_event_type(raw.get("event_type"))
    action = normalize_event_action(raw.get("action"))
    cards = normalize_event_cards(raw.get("cards"))
    amount = normalize_event_amount(raw.get("amount"))

    if event_type != "player_action":
        action = None
    if event_type != "card_update":
        cards = []
    if event_type not in {"player_action", "stack_update"}:
        amount = None

    return {
        "event_type": event_type,
        "action": action,
        "cards": cards,
        "amount": amount,
    }


def validate_expected_event(expected: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    normalized = normalize_expected_event(expected)
    event_type = normalized["event_type"]

    if event_type not in EVENT_TYPES:
        errors.append(f"unsupported event_type={event_type}")
    if event_type == "player_action" and normalized["action"] not in DECISION_ACTIONS:
        errors.append("player_action requires action in the decision label space")
    if event_type != "player_action" and normalized["action"] is not None:
        errors.append("non-action event must not carry an action label")
    if event_type == "card_update":
        if not normalized["cards"]:
            errors.append("card_update requires at least one card")
        invalid_cards = [card for card in normalized["cards"] if not CARD_PATTERN.match(card)]
        if invalid_cards:
            errors.append(f"invalid cards={invalid_cards}")
    if event_type != "card_update" and normalized["cards"]:
        errors.append("non-card event must not carry card labels")
    if event_type == "stack_update" and normalized["amount"] is None:
        errors.append("stack_update requires amount")
    if event_type == "unmatched":
        if normalized["action"] is not None or normalized["cards"] or normalized["amount"] is not None:
            errors.append("unmatched event must not carry action, cards, or amount")
    return ValidationResult(valid=not errors, errors=errors)


>>>>>>> 722cfd2 (LLM training validation simulation)
def validate_gold_row(row: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    if not row.get("id"):
        errors.append("row id is required")
    record = row.get("record")
    if not isinstance(record, dict):
        errors.append("record must be an object")
    else:
        if "frame_id" not in record:
            errors.append("record.frame_id is required")
        else:
            safe_int(record.get("frame_id"))
        if not str(record.get("event_name") or "").strip():
            errors.append("record.event_name is required")
        if "event_value" in record and not isinstance(record.get("event_value"), dict):
            errors.append("record.event_value must be an object when present")
    expected = row.get("expected")
    if not isinstance(expected, dict):
        errors.append("expected must be an object")
    else:
        expected_result = validate_expected_event(expected)
        errors.extend(expected_result.errors)
    return ValidationResult(valid=not errors, errors=errors)


def event_json_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": EVENT_SCHEMA_VERSION,
        "title": "Poker Event Extraction Gold Row",
        "type": "object",
        "required": ["id", "schema_version", "record", "expected", "group_id", "split"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "parent_id": {"type": "string", "minLength": 1},
            "group_id": {"type": "string", "minLength": 1},
            "split": {"type": "string", "enum": ["train", "valid", "test", "unassigned"]},
            "schema_version": {"type": "string", "const": EVENT_SCHEMA_VERSION},
            "noise": {
                "type": "object",
                "required": ["type", "severity"],
                "properties": {
                    "type": {"type": "string"},
                    "severity": {"type": "integer", "minimum": 0, "maximum": 3},
                },
                "additionalProperties": True,
            },
            "record": {
                "type": "object",
                "required": ["frame_id", "event_name", "event_value"],
                "properties": {
                    "frame_id": {"type": "integer"},
                    "event_name": {"type": "string"},
                    "object_type": {"type": "string"},
                    "event_value": {"type": "object"},
                },
                "additionalProperties": True,
            },
            "expected": {
                "type": "object",
                "required": ["event_type", "action", "cards", "amount"],
                "properties": {
                    "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
                    "action": {"anyOf": [{"type": "string", "enum": list(DECISION_ACTIONS)}, {"type": "null"}]},
                    "cards": {
                        "type": "array",
                        "items": {"type": "string", "pattern": CARD_PATTERN.pattern},
                    },
                    "amount": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": True,
    }


def schema_group_id(row: dict[str, Any]) -> str:
    record = row.get("record") if isinstance(row.get("record"), dict) else {}
    source_file = str(record.get("source_file") or row.get("source_file") or "").strip()
    if source_file:
        return source_file
    return str(row.get("parent_id") or row.get("id") or "unknown")

<<<<<<< HEAD
=======
>>>>>>> 249bf7f (LLM training/validation/simulation)
>>>>>>> 722cfd2 (LLM training validation simulation)
