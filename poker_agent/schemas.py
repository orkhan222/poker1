from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_ACTIONS = ("fold", "call", "check", "bet", "raise", "all_in")


def _cards_from_raw(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [card for card in raw.replace(",", " ").split() if card]
    return [str(card) for card in raw if str(card).strip()]


@dataclass
class PredictionRequest:
    position: str
    street: str = "preflop"
    hole_cards: list[str] = field(default_factory=list)
    board_cards: list[str] = field(default_factory=list)
    pot: float = 0.0
    to_call: float = 0.0
    stack: float = 0.0
    min_raise: float = 0.0
    player_count: int = 6
    betting_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PredictionRequest":
        return cls(
            position=str(raw.get("position") or raw.get("player_position") or "UNK"),
            street=str(raw.get("street") or "preflop").lower(),
            hole_cards=_cards_from_raw(raw.get("hole_cards", [])),
            board_cards=_cards_from_raw(raw.get("board_cards", [])),
            pot=float(raw.get("pot") or 0.0),
            to_call=float(raw.get("to_call") or 0.0),
            stack=float(raw.get("stack") or 0.0),
            min_raise=float(raw.get("min_raise") or 0.0),
            player_count=int(raw.get("player_count") or 6),
            betting_history=list(raw.get("betting_history") or raw.get("action_history") or []),
        )


@dataclass
class PredictionResponse:
    action: str
    probabilities: dict[str, float]
    confidence: float = 0.0
    bet_size: float | None = None
    wait_time_ms: int | None = None
    sizing_method: str | None = None
    timing_method: str | None = None
    model_status: str = "model"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        confidence = self.confidence or max(self.probabilities.values(), default=0.0)
        payload: dict[str, Any] = {
            "action": self.action,
            "probabilities": self.probabilities,
            "confidence": confidence,
            "model_status": self.model_status,
        }
        if self.bet_size is not None:
            payload["bet_size"] = self.bet_size
        if self.wait_time_ms is not None:
            payload["wait_time_ms"] = self.wait_time_ms
        if self.sizing_method:
            payload["sizing_method"] = self.sizing_method
        if self.timing_method:
            payload["timing_method"] = self.timing_method
        if self.warnings:
            payload["warnings"] = self.warnings
        return payload
