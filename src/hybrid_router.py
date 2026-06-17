from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from poker_agent.event_normalization.parser import DeterministicParser
from poker_agent.event_normalization.schema import Event, ExtractionResult, model_dump_event


@dataclass(frozen=True)
class HybridDecision:
    result: ExtractionResult
    route: str
    parser_event: Event
    model_attempted: bool


def parser_event_is_usable(event: Event, *, min_confidence: float) -> bool:
    if event.event_type == "unmatched":
        return False
    if event.confidence < min_confidence:
        return False
    if event.event_type == "player_action" and event.action is None:
        return False
    if event.event_type == "card_event" and not event.normalized_cards:
        return False
    if event.event_type == "pot_event" and event.amount is None:
        return False
    return True


def model_result_is_usable(result: ExtractionResult, *, min_confidence: float) -> bool:
    if not result.schema_valid:
        return False
    if result.event.event_type == "unmatched":
        return False
    if result.event.confidence < min_confidence:
        return False
    if result.event.event_type == "player_action" and result.event.action is None:
        return False
    if result.event.event_type == "card_event" and not result.event.normalized_cards:
        return False
    if result.event.event_type == "pot_event" and result.event.amount is None:
        return False
    return True


class HybridEventRouter:
    def __init__(
        self,
        *,
        parser: DeterministicParser,
        model_extractor: Callable[[str], ExtractionResult] | None,
        parser_min_confidence: float = 0.80,
        model_min_confidence: float = 0.50,
        allow_model_override: bool = False,
    ) -> None:
        self.parser = parser
        self.model_extractor = model_extractor
        self.parser_min_confidence = parser_min_confidence
        self.model_min_confidence = model_min_confidence
        self.allow_model_override = allow_model_override

    def extract(self, raw_text: str) -> HybridDecision:
        parser_event = self.parser.parse(raw_text)
        parser_result = ExtractionResult(
            event=parser_event,
            raw_output=self._raw_parser_output(parser_event),
            schema_valid=True,
        )
        parser_usable = parser_event_is_usable(parser_event, min_confidence=self.parser_min_confidence)
        if parser_usable and not self.allow_model_override:
            return HybridDecision(
                result=parser_result,
                route="parser",
                parser_event=parser_event,
                model_attempted=False,
            )

        model_result = self.model_extractor(raw_text) if self.model_extractor is not None else None
        if model_result is not None and model_result_is_usable(model_result, min_confidence=self.model_min_confidence):
            return HybridDecision(
                result=model_result,
                route="model_fallback",
                parser_event=parser_event,
                model_attempted=True,
            )

        if parser_usable:
            return HybridDecision(
                result=parser_result,
                route="parser_after_model_reject",
                parser_event=parser_event,
                model_attempted=model_result is not None,
            )
        return HybridDecision(
            result=parser_result,
            route="unmatched",
            parser_event=parser_event,
            model_attempted=model_result is not None,
        )

    def _raw_parser_output(self, event: Event) -> str:
        import json

        payload = model_dump_event(event)
        payload["cards"] = payload.get("cards") or None
        return json.dumps(payload, sort_keys=True)

