from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from poker_agent.event_normalization.parser import DeterministicParser
from poker_agent.event_normalization.schema import Event, ExtractionResult, model_dump_event


ModelExtractor = Callable[[str], ExtractionResult]


@dataclass(frozen=True)
class AgentConfig:
    architecture: str = "hybrid_parser_qlora"
    parser_min_confidence: float = 0.80
    model_min_confidence: float = 0.50
    allow_model_override: bool = False


@dataclass(frozen=True)
class AgentPrediction:
    event: Event
    architecture: str
    route: str
    schema_valid: bool
    raw_output: str
    errors: list[str]
    latency_ms: float
    parser_event: Event | None = None
    model_attempted: bool = False
    model_used: bool = False

    def to_record(self) -> dict[str, Any]:
        return {
            "event": model_dump_event(self.event),
            "architecture": self.architecture,
            "route": self.route,
            "schema_valid": self.schema_valid,
            "raw_output": self.raw_output,
            "errors": list(self.errors),
            "latency_ms": self.latency_ms,
            "parser_event": None if self.parser_event is None else model_dump_event(self.parser_event),
            "model_attempted": self.model_attempted,
            "model_used": self.model_used,
        }


def event_is_complete(event: Event, *, min_confidence: float) -> bool:
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


def result_is_complete(result: ExtractionResult, *, min_confidence: float) -> bool:
    if not result.schema_valid:
        return False
    return event_is_complete(result.event, min_confidence=min_confidence)


class EventNormalizationAgent:
    """Production event-normalization agent for OCR and dealer-log records."""

    def __init__(
        self,
        *,
        config: AgentConfig | None = None,
        parser: DeterministicParser | None = None,
        model_extractor: ModelExtractor | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.parser = parser or DeterministicParser(ocr_profile="full")
        self.model_extractor = model_extractor

    def normalize(self, raw_text: str) -> AgentPrediction:
        started = time.perf_counter()
        architecture = self.config.architecture
        if architecture == "deterministic_parser":
            result = self._parser_result(raw_text)
            return self._prediction(
                result=result,
                route="parser",
                started=started,
                parser_event=result.event,
            )
        if architecture == "qlora_extractor":
            result = self._model_result(raw_text)
            return self._prediction(
                result=result,
                route="model",
                started=started,
                model_attempted=True,
                model_used=result_is_complete(result, min_confidence=self.config.model_min_confidence),
            )
        if architecture == "hybrid_parser_qlora":
            return self._hybrid_prediction(raw_text, started)
        raise ValueError(f"Unsupported event-normalization architecture: {architecture}")

    def _hybrid_prediction(self, raw_text: str, started: float) -> AgentPrediction:
        parser_result = self._parser_result(raw_text)
        parser_usable = event_is_complete(
            parser_result.event,
            min_confidence=self.config.parser_min_confidence,
        )
        if parser_usable and not self.config.allow_model_override:
            return self._prediction(
                result=parser_result,
                route="parser",
                started=started,
                parser_event=parser_result.event,
                model_attempted=False,
                model_used=False,
            )

        model_result = self._model_result(raw_text)
        if result_is_complete(model_result, min_confidence=self.config.model_min_confidence):
            return self._prediction(
                result=model_result,
                route="model_fallback",
                started=started,
                parser_event=parser_result.event,
                model_attempted=True,
                model_used=True,
            )

        if parser_usable:
            return self._prediction(
                result=parser_result,
                route="parser_after_model_reject",
                started=started,
                parser_event=parser_result.event,
                model_attempted=True,
                model_used=False,
            )

        return self._prediction(
            result=parser_result,
            route="unmatched",
            started=started,
            parser_event=parser_result.event,
            model_attempted=True,
            model_used=False,
        )

    def _parser_result(self, raw_text: str) -> ExtractionResult:
        event = self.parser.parse(raw_text)
        payload = model_dump_event(event)
        payload["cards"] = payload.get("cards") or None
        return ExtractionResult(
            event=event,
            raw_output=json.dumps(payload, sort_keys=True),
            schema_valid=True,
        )

    def _model_result(self, raw_text: str) -> ExtractionResult:
        if self.model_extractor is None:
            return ExtractionResult(
                event=Event(event_type="unmatched", confidence=0.0),
                raw_output="",
                schema_valid=False,
                errors=["model extractor is not configured"],
            )
        return self.model_extractor(raw_text)

    def _prediction(
        self,
        *,
        result: ExtractionResult,
        route: str,
        started: float,
        parser_event: Event | None = None,
        model_attempted: bool = False,
        model_used: bool = False,
    ) -> AgentPrediction:
        return AgentPrediction(
            event=result.event,
            architecture=self.config.architecture,
            route=route,
            schema_valid=result.schema_valid,
            raw_output=result.raw_output,
            errors=list(result.errors),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            parser_event=parser_event,
            model_attempted=model_attempted,
            model_used=model_used,
        )

