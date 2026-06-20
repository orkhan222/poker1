from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_agent.features import request_to_features
from poker_agent.schemas import PredictionRequest


ZERO_SIZE_ACTIONS = {"fold", "check"}


@dataclass(frozen=True)
class ActionPlan:
    bet_size: float
    wait_time_ms: int
    sizing_method: str
    timing_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bet_size": self.bet_size,
            "wait_time_ms": self.wait_time_ms,
            "sizing_method": self.sizing_method,
            "timing_method": self.timing_method,
        }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def round_chips(value: float) -> float:
    return round(max(0.0, value), 2)


def legal_min_raise(request: PredictionRequest) -> float:
    if request.min_raise > 0:
        return request.min_raise
    if request.to_call > 0:
        return max(request.to_call * 2.0, 1.0)
    return max(request.pot * 0.5, 1.0)


def calibrated_aggressive_size(
    request: PredictionRequest,
    action: str,
    *,
    strength: float,
    confidence: float,
) -> float:
    stack = max(float(request.stack), 0.0)
    pot = max(float(request.pot), 0.0)
    minimum = min(stack, legal_min_raise(request))
    if stack <= 0.0:
        return 0.0
    if action == "all_in":
        return stack

    pressure_score = clamp((0.62 * strength + 0.38 * confidence - 0.68) / 0.32, 0.0, 1.0)
    if pressure_score <= 0.0:
        return minimum

    extra_fraction = 0.10 + 0.18 * strength + 0.08 * confidence
    raw_size = minimum + pot * extra_fraction * pressure_score
    stack_cap = stack * (0.24 + 0.18 * pressure_score)
    return min(stack, max(minimum, min(raw_size, stack_cap)))


def estimate_bet_size(
    request: PredictionRequest,
    action: str,
    *,
    confidence: float,
) -> tuple[float, str]:
    stack = max(float(request.stack), 0.0)
    to_call = max(float(request.to_call), 0.0)
    confidence = clamp(float(confidence), 0.0, 1.0)
    features = request_to_features(request)
    strength = clamp(float(features.get("strength_proxy", 0.0)), 0.0, 1.0)

    if stack <= 0.0 or action in ZERO_SIZE_ACTIONS:
        return 0.0, "no_chip_commitment"
    if action == "call":
        return round_chips(min(stack, to_call)), "call_price"
    if action == "bet":
        if to_call > 0.0:
            return round_chips(min(stack, to_call)), "bet_mapped_to_call_price"
        capped = calibrated_aggressive_size(
            request,
            action,
            strength=strength,
            confidence=confidence,
        )
        return round_chips(capped), "pot_fraction_bet"
    if action in {"raise", "all_in"}:
        capped = calibrated_aggressive_size(
            request,
            action,
            strength=strength,
            confidence=confidence,
        )
        return round_chips(capped), "pressure_raise"
    return 0.0, "unsupported_action"


def estimate_wait_time_ms(
    request: PredictionRequest,
    action: str,
    *,
    confidence: float,
    processing_time_ms: float = 0.0,
) -> tuple[int, str]:
    confidence = clamp(float(confidence), 0.0, 1.0)
    history_depth = min(len(request.betting_history), 12)
    street_complexity = {"preflop": 0, "flop": 180, "turn": 260, "river": 340}.get(request.street, 120)
    action_complexity = {"fold": 120, "check": 100, "call": 180, "bet": 320, "raise": 420, "all_in": 520}.get(
        action,
        200,
    )
    uncertainty_penalty = int((1.0 - confidence) * 650)
    history_penalty = history_depth * 45
    base_ms = 550 + street_complexity + action_complexity + uncertainty_penalty + history_penalty
    minimum_ms = int(max(0.0, processing_time_ms) + 75)
    return int(clamp(max(base_ms, minimum_ms), 250, 8000)), "complexity_calibrated"


def build_action_plan(
    request: PredictionRequest,
    action: str,
    *,
    confidence: float,
    processing_time_ms: float = 0.0,
) -> ActionPlan:
    bet_size, sizing_method = estimate_bet_size(request, action, confidence=confidence)
    wait_time_ms, timing_method = estimate_wait_time_ms(
        request,
        action,
        confidence=confidence,
        processing_time_ms=processing_time_ms,
    )
    return ActionPlan(
        bet_size=bet_size,
        wait_time_ms=wait_time_ms,
        sizing_method=sizing_method,
        timing_method=timing_method,
    )
