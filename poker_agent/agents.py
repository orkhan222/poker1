from __future__ import annotations

from pathlib import Path
from typing import Any

from poker_agent.action_planning import build_action_plan
from poker_agent.features import request_to_features
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest, PredictionResponse


class RuleBasedAgent:
    def predict(self, request: PredictionRequest, warnings: list[str] | None = None) -> PredictionResponse:
        strength = request_to_features(request)["strength_proxy"]
        if request.to_call <= 0 and strength < 0.45:
            probabilities = {"check": 0.72, "bet": 0.18, "fold": 0.04, "call": 0.04, "raise": 0.02}
        elif strength >= 0.75:
            probabilities = {"raise": 0.52, "call": 0.24, "bet": 0.18, "check": 0.04, "fold": 0.02}
        elif strength >= 0.48:
            probabilities = {"call": 0.52, "raise": 0.18, "check": 0.16, "fold": 0.10, "bet": 0.04}
        else:
            probabilities = {"fold": 0.62, "call": 0.20, "check": 0.12, "raise": 0.04, "bet": 0.02}
        action = max(probabilities, key=probabilities.get)
        confidence = max(probabilities.values(), default=0.0)
        plan = build_action_plan(request, action, confidence=confidence)
        return PredictionResponse(
            action=action,
            probabilities=probabilities,
            confidence=confidence,
            bet_size=plan.bet_size,
            wait_time_ms=plan.wait_time_ms,
            sizing_method=plan.sizing_method,
            timing_method=plan.timing_method,
            model_status="rule_based",
            warnings=warnings or [],
        )


class MissingCardFallbackAgent:
    """Conservative context policy for out-of-distribution missing-card requests."""

    def predict(self, request: PredictionRequest, warnings: list[str] | None = None) -> PredictionResponse:
        pot_odds = request.to_call / (request.pot + request.to_call) if request.pot + request.to_call > 0 else 0.0
        no_price_to_continue = request.to_call <= 0
        low_price = 0.0 < pot_odds <= 0.18
        medium_price = 0.18 < pot_odds <= 0.32

        if no_price_to_continue:
            probabilities = {"check": 0.66, "bet": 0.14, "fold": 0.08, "call": 0.08, "raise": 0.04}
        elif low_price:
            probabilities = {"call": 0.46, "fold": 0.34, "raise": 0.10, "bet": 0.06, "check": 0.04}
        elif medium_price:
            probabilities = {"fold": 0.52, "call": 0.34, "raise": 0.07, "bet": 0.04, "check": 0.03}
        else:
            probabilities = {"fold": 0.72, "call": 0.18, "raise": 0.04, "bet": 0.03, "check": 0.03}

        action = max(probabilities, key=probabilities.get)
        confidence = max(probabilities.values(), default=0.0)
        plan = build_action_plan(request, action, confidence=confidence)
        return PredictionResponse(
            action=action,
            probabilities=probabilities,
            confidence=confidence,
            bet_size=plan.bet_size,
            wait_time_ms=plan.wait_time_ms,
            sizing_method=plan.sizing_method,
            timing_method=plan.timing_method,
            model_status="missing_card_fallback",
            warnings=warnings or [],
        )


class MLPolicyAgent:
    def __init__(self, model: Any):
        self.model = model
        self.missing_card_fallback = MissingCardFallbackAgent()

    @classmethod
    def from_path(cls, path: Path) -> "MLPolicyAgent":
        return cls(load_policy(path))

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        warnings: list[str] = []
        metadata = getattr(self.model, "metadata", {}) or {}
        trained_missing_mode = str(metadata.get("missing_hole_cards", "unknown"))
        if len(request.hole_cards) < 2 and trained_missing_mode == "drop":
            warnings.append(
                "Hole cards are missing, while the loaded model was trained with missing-card rows dropped. "
                "Using conservative context fallback instead of out-of-distribution model inference."
            )
            return self.missing_card_fallback.predict(request, warnings=warnings)

        action, probabilities = self.model.predict_from_features(request_to_features(request))
        confidence = max(probabilities.values(), default=0.0)
        plan = build_action_plan(request, action, confidence=confidence)
        return PredictionResponse(
            action=action,
            probabilities=probabilities,
            confidence=confidence,
            bet_size=plan.bet_size,
            wait_time_ms=plan.wait_time_ms,
            sizing_method=plan.sizing_method,
            timing_method=plan.timing_method,
            model_status=str(metadata.get("policy", "model")),
            warnings=warnings,
        )
