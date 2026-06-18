from __future__ import annotations

from poker_agent.event_normalization.agent import AgentConfig, EventNormalizationAgent
from poker_agent.event_normalization.schema import ExtractionResult, build_event


def test_deterministic_agent_extracts_player_action() -> None:
    agent = EventNormalizationAgent(config=AgentConfig(architecture="deterministic_parser"))

    prediction = agent.normalize("Plyr3 ra1se $4.50")

    assert prediction.route == "parser"
    assert prediction.schema_valid is True
    assert prediction.event.event_type == "player_action"
    assert prediction.event.player == "Player3"
    assert prediction.event.action == "raise"
    assert prediction.event.amount == 4.5


def test_hybrid_agent_does_not_load_model_for_high_confidence_parser_event() -> None:
    def failing_model(_: str) -> ExtractionResult:
        raise AssertionError("model fallback should not be called for parser-complete records")

    agent = EventNormalizationAgent(
        config=AgentConfig(architecture="hybrid_parser_qlora"),
        model_extractor=failing_model,
    )

    prediction = agent.normalize("Player2 calls $1.00")

    assert prediction.route == "parser"
    assert prediction.model_attempted is False
    assert prediction.model_used is False
    assert prediction.event.action == "call"


def test_hybrid_agent_uses_model_fallback_for_unmatched_parser_event() -> None:
    def model(_: str) -> ExtractionResult:
        return ExtractionResult(
            event=build_event(
                {
                    "event_type": "player_action",
                    "player": "Player9",
                    "action": "fold",
                    "confidence": 0.91,
                }
            ),
            raw_output='{"event_type":"player_action","player":"Player9","action":"fold","confidence":0.91}',
            schema_valid=True,
        )

    agent = EventNormalizationAgent(
        config=AgentConfig(
            architecture="hybrid_parser_qlora",
            parser_min_confidence=0.80,
            model_min_confidence=0.50,
        ),
        model_extractor=model,
    )

    prediction = agent.normalize("unreadable dealer fragment ###")

    assert prediction.route == "model_fallback"
    assert prediction.model_attempted is True
    assert prediction.model_used is True
    assert prediction.event.event_type == "player_action"
    assert prediction.event.player == "Player9"
    assert prediction.event.action == "fold"

