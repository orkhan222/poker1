from __future__ import annotations

import bisect
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from poker_agent.action_planning import build_action_plan
from poker_agent.agents import RuleBasedAgent
from poker_agent.features import request_to_features
from poker_agent.holdem_self_play import evaluate_seven_cards, run_holdem_self_play
from poker_agent.schemas import PredictionRequest


ACTION_ORDER = ("fold", "call", "check", "bet", "raise")
RANKS = "23456789TJQKA"
SUITS = "SHDC"
POSITIONS = ("SB", "BB", "UTG", "MP", "CO", "BTN")
STREETS = ("preflop", "flop", "turn", "river")
TIMING_BUCKETS = ("instant", "short", "medium", "long")
BET_SIZE_BUCKETS = ("micro", "small", "medium", "large")
CHIP_COMMITMENT_ACTIONS = {"call", "bet", "raise", "all_in"}
STACK_ACTION_MATCH_WINDOW_FRAMES = 160.0
RECONSTRUCTED_CONTEXT_FEATURES = (
    "call_price_ratio",
    "hero_commitment_ratio",
    "players_acted_ratio",
    "street_action_count",
    "street_aggression_ratio",
    "table_commitment_pressure",
)


@dataclass(frozen=True)
class SimulationConfig:
    hands: int = 1000
    seed: int = 42
    player_count: int = 6


def normalize_action(action: str) -> str:
    text = str(action or "").strip().lower()
    if text == "all_in":
        return "raise"
    if text in ACTION_ORDER:
        return text
    return "fold"


def normalize_distribution(counts: Counter[str] | dict[str, float]) -> dict[str, float]:
    values = {action: float(counts.get(action, 0.0)) for action in ACTION_ORDER}
    total = sum(values.values())
    if total <= 0.0:
        return {action: 1.0 / len(ACTION_ORDER) for action in ACTION_ORDER}
    return {action: value / total for action, value in values.items()}


def total_variation_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return 0.5 * sum(abs(left.get(action, 0.0) - right.get(action, 0.0)) for action in ACTION_ORDER)


def jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    midpoint = {action: 0.5 * (left.get(action, 0.0) + right.get(action, 0.0)) for action in ACTION_ORDER}
    return 0.5 * _kl_divergence(left, midpoint) + 0.5 * _kl_divergence(right, midpoint)


def _kl_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    total = 0.0
    for action in ACTION_ORDER:
        p = max(left.get(action, 0.0), 0.0)
        q = max(right.get(action, 0.0), 1e-12)
        if p > 0.0:
            total += p * math.log2(p / q)
    return total


def normalize_distribution_for_keys(
    counts: Counter[str] | dict[str, float],
    keys: Iterable[str],
) -> dict[str, float]:
    ordered_keys = tuple(keys)
    values = {key: float(counts.get(key, 0.0)) for key in ordered_keys}
    total = sum(values.values())
    if total <= 0.0:
        return {key: 1.0 / len(ordered_keys) for key in ordered_keys}
    return {key: value / total for key, value in values.items()}


def jensen_shannon_for_keys(
    left: dict[str, float],
    right: dict[str, float],
    keys: Iterable[str],
) -> float:
    ordered_keys = tuple(keys)
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in ordered_keys}
    return 0.5 * _kl_divergence_for_keys(left, midpoint, ordered_keys) + 0.5 * _kl_divergence_for_keys(
        right,
        midpoint,
        ordered_keys,
    )


def total_variation_for_keys(
    left: dict[str, float],
    right: dict[str, float],
    keys: Iterable[str],
) -> float:
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _kl_divergence_for_keys(left: dict[str, float], right: dict[str, float], keys: Iterable[str]) -> float:
    total = 0.0
    for key in keys:
        p = max(left.get(key, 0.0), 0.0)
        q = max(right.get(key, 0.0), 1e-12)
        if p > 0.0:
            total += p * math.log2(p / q)
    return total


def prediction_action(model: Any, features: dict[str, float]) -> str:
    action, probabilities = model.predict_from_features(features)
    if probabilities:
        action = max(probabilities, key=probabilities.get)
    return normalize_action(action)


def prediction_actions(model: Any, feature_rows: list[dict[str, float]]) -> list[str]:
    if hasattr(model, "predict_batch_from_features"):
        actions: list[str] = []
        for action, probabilities in model.predict_batch_from_features(feature_rows):
            selected = max(probabilities, key=probabilities.get) if probabilities else action
            actions.append(normalize_action(selected))
        return actions
    return [prediction_action(model, features) for features in feature_rows]


def softmax(scores: dict[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}
    temperature = max(float(temperature), 1e-6)
    max_score = max(scores.values())
    values = {key: math.exp((value - max_score) / temperature) for key, value in scores.items()}
    total = sum(values.values()) or 1.0
    return {key: value / total for key, value in values.items()}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def holdem_decision_strength(features: dict[str, float]) -> float:
    street_index = float(features.get("street_index", 0.0))
    preflop = max(
        float(features.get("strength_proxy", 0.0)),
        float(features.get("preflop_bucket_score", 0.0)),
    )
    if street_index <= 0.0:
        return clamp(preflop, 0.0, 1.0)

    made = float(features.get("made_hand_score", 0.0))
    draw = max(
        0.35 * float(features.get("straight_draw_score", 0.0)),
        0.32 * float(features.get("flush_draw_pressure", 0.0)),
    )
    pair_floor = 0.46 if features.get("top_pair_or_better", 0.0) > 0 else 0.0
    overpair_floor = 0.58 if features.get("hole_overpair", 0.0) > 0 else 0.0
    board_wetness = float(features.get("board_wetness", 0.0))
    raw = max(made, pair_floor, overpair_floor) + draw + 0.22 * preflop
    if made < 0.22:
        raw -= 0.10 * board_wetness
    return clamp(raw, 0.0, 1.0)


def expected_action_ev(features: dict[str, float], action: str) -> float:
    action = normalize_action(action)
    strength = holdem_decision_strength(features)
    pot = max(float(features.get("pot", 0.0)), 1.0)
    to_call = max(float(features.get("to_call", 0.0)), 0.0)
    stack = max(float(features.get("stack", 0.0)), 1.0)
    min_raise = max(float(features.get("min_raise", 0.0)), 1.0)
    pressure = min(to_call / pot, 1.0)
    street_index = float(features.get("street_index", 0.0))
    player_count = max(float(features.get("player_count", 6.0)), 2.0)
    multiway_penalty = min(0.18, 0.035 * max(player_count - 2.0, 0.0))
    aggression = min(float(features.get("hist_aggression_ratio", 0.0)), 1.0)
    win_probability = clamp(
        0.08 + 0.86 * strength - 0.16 * pressure - multiway_penalty + 0.025 * street_index,
        0.03,
        0.94,
    )

    if action == "fold":
        return 0.0 if to_call > 0 else -0.02 * pot
    if action == "check":
        if to_call > 0:
            return -10.0 * pot
        return win_probability * (0.10 * pot) - (1.0 - win_probability) * (0.025 * pot)
    if action == "call":
        if to_call <= 0:
            return win_probability * (0.04 * pot) - (1.0 - win_probability) * (0.02 * pot)
        return win_probability * pot - (1.0 - win_probability) * to_call
    if action == "bet":
        if to_call > 0:
            return -10.0 * pot
        bet_size = min(max(min_raise, pot * (0.38 + 0.20 * strength)), stack * 0.22)
        fold_equity = clamp(0.08 + 0.25 * strength + 0.06 * aggression - 0.025 * max(player_count - 2.0, 0.0), 0.03, 0.46)
        called_ev = win_probability * (pot + bet_size) - (1.0 - win_probability) * bet_size
        if strength < 0.36:
            called_ev -= 0.18 * bet_size
        return fold_equity * pot + (1.0 - fold_equity) * called_ev

    raise_size = min(max(min_raise, pot * (0.55 + 0.25 * strength)), stack * 0.30)
    fold_equity = clamp(0.10 + 0.30 * strength - 0.16 * pressure - 0.025 * max(player_count - 2.0, 0.0), 0.02, 0.52)
    called_ev = win_probability * (pot + raise_size) - (1.0 - win_probability) * raise_size
    if to_call > 0 and win_probability < max(0.35, pressure + 0.12):
        called_ev -= 0.24 * raise_size
    return fold_equity * pot + (1.0 - fold_equity) * called_ev


def ev_calibrated_probabilities(
    features: dict[str, float],
    model_probabilities: dict[str, float],
    *,
    ev_weight: float = 0.72,
    temperature: float = 1.6,
    min_ev_gain: float = 0.02,
) -> dict[str, float]:
    ev_scores = {action: expected_action_ev(features, action) for action in ACTION_ORDER}
    pot_scale = max(float(features.get("pot", 0.0)), 1.0)
    normalized_ev_scores = {action: score / pot_scale for action, score in ev_scores.items()}
    ev_probabilities = softmax(normalized_ev_scores, temperature=temperature)
    model = {action: max(float(model_probabilities.get(action, 0.0)), 0.0) for action in ACTION_ORDER}
    model_total = sum(model.values()) or 1.0
    model = {action: value / model_total for action, value in model.items()}

    weight = min(max(ev_weight, 0.0), 1.0)
    blended = {
        action: (1.0 - weight) * model.get(action, 0.0) + weight * ev_probabilities.get(action, 0.0)
        for action in ACTION_ORDER
    }
    model_action = max(model, key=model.get)
    ev_action = max(ev_scores, key=ev_scores.get)
    if ev_action != model_action and ev_scores[ev_action] - ev_scores[model_action] >= min_ev_gain * pot_scale:
        blended[ev_action] = max(blended[ev_action], min(0.55, blended[model_action] + 0.08))

    total = sum(blended.values()) or 1.0
    return {action: value / total for action, value in blended.items()}


def rule_based_reference_probabilities(features: dict[str, float]) -> dict[str, float]:
    strength = float(features.get("strength_proxy", 0.0))
    to_call = float(features.get("to_call", 0.0))
    if to_call <= 0.0 and strength < 0.45:
        return {"check": 0.72, "bet": 0.18, "fold": 0.04, "call": 0.04, "raise": 0.02}
    if strength >= 0.75:
        return {"raise": 0.52, "call": 0.24, "bet": 0.18, "check": 0.04, "fold": 0.02}
    if strength >= 0.48:
        return {"call": 0.52, "raise": 0.18, "check": 0.16, "fold": 0.10, "bet": 0.04}
    return {"fold": 0.62, "call": 0.20, "check": 0.12, "raise": 0.04, "bet": 0.02}


def holdem_aware_probabilities(features: dict[str, float]) -> dict[str, float]:
    probabilities = rule_based_reference_probabilities(features)
    street_index = float(features.get("street_index", 0.0))
    if street_index <= 0.0:
        return probabilities

    strength = holdem_decision_strength(features)
    to_call = float(features.get("to_call", 0.0))
    if to_call > 0.0 and strength <= 0.25:
        return {"fold": 0.76, "call": 0.18, "check": 0.02, "bet": 0.01, "raise": 0.03}
    if strength >= 0.72:
        if to_call > 0.0:
            return {"fold": 0.02, "call": 0.48, "check": 0.02, "bet": 0.02, "raise": 0.46}
        return {"fold": 0.01, "call": 0.03, "check": 0.50, "bet": 0.08, "raise": 0.38}
    return probabilities


def holdem_request_probabilities(request: PredictionRequest) -> dict[str, float]:
    features = request_to_features(request)
    probabilities = holdem_aware_probabilities(features)
    if request.street == "preflop" or len(request.hole_cards) < 2 or len(request.board_cards) < 3:
        return probabilities

    rank_class, kickers = evaluate_seven_cards(request.hole_cards + request.board_cards)
    to_call = max(float(request.to_call), 0.0)
    pot = max(float(request.pot), 1.0)
    pot_odds = to_call / (pot + to_call) if pot + to_call > 0 else 0.0
    top_pair_or_better = float(features.get("top_pair_or_better", 0.0)) > 0.0
    overpair = float(features.get("hole_overpair", 0.0)) > 0.0
    draw_pressure = max(
        float(features.get("straight_draw_score", 0.0)),
        float(features.get("flush_draw_pressure", 0.0)),
    )

    if to_call > 0.0:
        if rank_class >= 3:
            return {"fold": 0.02, "call": 0.38, "check": 0.01, "bet": 0.01, "raise": 0.58}
        if rank_class == 2:
            return {"fold": 0.04, "call": 0.58, "check": 0.01, "bet": 0.01, "raise": 0.36}
        if rank_class == 1 and (top_pair_or_better or overpair):
            if pot_odds <= 0.30:
                return {"fold": 0.10, "call": 0.66, "check": 0.02, "bet": 0.01, "raise": 0.21}
            return {"fold": 0.46, "call": 0.42, "check": 0.02, "bet": 0.01, "raise": 0.09}
        if draw_pressure >= 0.75 and pot_odds <= 0.22:
            return {"fold": 0.22, "call": 0.60, "check": 0.02, "bet": 0.01, "raise": 0.15}
        return {"fold": 0.78, "call": 0.17, "check": 0.02, "bet": 0.01, "raise": 0.02}

    if rank_class >= 4:
        return {"fold": 0.01, "call": 0.02, "check": 0.44, "bet": 0.10, "raise": 0.43}
    if rank_class >= 2:
        return {"fold": 0.02, "call": 0.03, "check": 0.58, "bet": 0.08, "raise": 0.29}
    if rank_class == 1 and (top_pair_or_better or overpair):
        return {"fold": 0.03, "call": 0.04, "check": 0.68, "bet": 0.06, "raise": 0.19}
    return probabilities


class EVCalibratedPolicy:
    """Wrap a supervised policy with a deterministic EV-aware action selector."""

    def __init__(
        self,
        base_model: Any,
        *,
        ev_weight: float = 0.86,
        temperature: float = 1.6,
        min_ev_gain: float = 0.02,
    ) -> None:
        self.base_model = base_model
        self.ev_weight = ev_weight
        self.temperature = temperature
        self.min_ev_gain = min_ev_gain
        self.metadata = dict(getattr(base_model, "metadata", {}) or {})
        self.metadata["strategy_selector"] = "ev_calibrated"

    def predict_proba_from_features(self, raw_features: dict[str, float]) -> dict[str, float]:
        probabilities = self.base_model.predict_proba_from_features(raw_features)
        return ev_calibrated_probabilities(
            raw_features,
            probabilities,
            ev_weight=self.ev_weight,
            temperature=self.temperature,
            min_ev_gain=self.min_ev_gain,
        )

    def predict_from_features(self, raw_features: dict[str, float]) -> tuple[str, dict[str, float]]:
        probabilities = self.predict_proba_from_features(raw_features)
        return max(probabilities, key=probabilities.get), probabilities

    def predict_batch_from_features(
        self,
        feature_rows: list[dict[str, float]],
    ) -> list[tuple[str, dict[str, float]]]:
        return [self.predict_from_features(features) for features in feature_rows]


def reconstructed_context_score(features: dict[str, float]) -> float:
    present = sum(1 for name in RECONSTRUCTED_CONTEXT_FEATURES if name in features)
    return present / len(RECONSTRUCTED_CONTEXT_FEATURES)


class DeploymentGatedPolicy:
    """Use supervised imitation on reconstructed histories and EV overlay on OOD smoke contexts."""

    def __init__(
        self,
        base_model: Any,
        *,
        ev_weight: float = 0.86,
        temperature: float = 1.6,
        min_ev_gain: float = 0.02,
        min_reconstructed_context_score: float = 0.67,
    ) -> None:
        self.base_model = base_model
        self.ev_weight = ev_weight
        self.temperature = temperature
        self.min_ev_gain = min_ev_gain
        self.min_reconstructed_context_score = min_reconstructed_context_score
        self.metadata = dict(getattr(base_model, "metadata", {}) or {})
        self.metadata["strategy_selector"] = "deployment_gated"
        self.metadata["min_reconstructed_context_score"] = min_reconstructed_context_score

    def predict_proba_from_features(self, raw_features: dict[str, float]) -> dict[str, float]:
        probabilities = self.base_model.predict_proba_from_features(raw_features)
        if reconstructed_context_score(raw_features) >= self.min_reconstructed_context_score:
            return probabilities
        return holdem_aware_probabilities(raw_features)

    def predict_from_features(self, raw_features: dict[str, float]) -> tuple[str, dict[str, float]]:
        probabilities = self.predict_proba_from_features(raw_features)
        return max(probabilities, key=probabilities.get), probabilities

    def predict_request(self, request: PredictionRequest) -> tuple[str, dict[str, float]]:
        features = request_to_features(request)
        if reconstructed_context_score(features) >= self.min_reconstructed_context_score:
            return self.predict_from_features(features)
        probabilities = holdem_request_probabilities(request)
        return max(probabilities, key=probabilities.get), probabilities

    def predict_batch_from_features(
        self,
        feature_rows: list[dict[str, float]],
    ) -> list[tuple[str, dict[str, float]]]:
        return [self.predict_from_features(features) for features in feature_rows]


def bucket_name(features: dict[str, float]) -> str:
    street = "preflop" if features.get("street_index", 0.0) == 0.0 else "postflop"
    facing = "facing_bet" if features.get("facing_bet_or_raise", features.get("has_call", 0.0)) > 0 else "not_facing_bet"
    strength = float(features.get("strength_proxy", 0.0))
    if strength >= 0.66:
        strength_bucket = "strong"
    elif strength >= 0.40:
        strength_bucket = "medium"
    else:
        strength_bucket = "weak_or_missing"
    return f"{street}:{facing}:{strength_bucket}"


def evaluate_human_likeness(
    model: Any,
    examples: list[tuple[dict[str, float], str]],
    *,
    max_js_divergence: float = 0.08,
    max_total_variation: float = 0.20,
    min_bucket_examples: int = 50,
    timing_bet_size: dict[str, Any] | None = None,
) -> dict[str, Any]:
    human_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    bucket_human: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_model: dict[str, Counter[str]] = defaultdict(Counter)

    feature_rows = [features for features, _label in examples]
    model_actions = prediction_actions(model, feature_rows)

    for (features, label), model_action in zip(examples, model_actions):
        human_action = normalize_action(label)
        bucket = bucket_name(features)
        human_counts[human_action] += 1
        model_counts[model_action] += 1
        bucket_human[bucket][human_action] += 1
        bucket_model[bucket][model_action] += 1

    human_distribution = normalize_distribution(human_counts)
    model_distribution = normalize_distribution(model_counts)
    js = jensen_shannon_divergence(human_distribution, model_distribution)
    tv = total_variation_distance(human_distribution, model_distribution)

    bucket_metrics: dict[str, dict[str, Any]] = {}
    weighted_bucket_js = 0.0
    bucket_weight = 0
    for bucket, counts in sorted(bucket_human.items()):
        support = sum(counts.values())
        if support < min_bucket_examples:
            continue
        human_bucket = normalize_distribution(counts)
        model_bucket = normalize_distribution(bucket_model[bucket])
        bucket_js = jensen_shannon_divergence(human_bucket, model_bucket)
        bucket_tv = total_variation_distance(human_bucket, model_bucket)
        weighted_bucket_js += bucket_js * support
        bucket_weight += support
        bucket_metrics[bucket] = {
            "support": support,
            "js_divergence": bucket_js,
            "total_variation": bucket_tv,
            "human_distribution": human_bucket,
            "model_distribution": model_bucket,
        }

    action_distribution_status = "PASS" if js <= max_js_divergence and tv <= max_total_variation else "FAIL"
    timing_bet_size = timing_bet_size or unavailable_timing_bet_size_report()
    timing_and_bet_size_status = str(timing_bet_size.get("status", "NOT_AVAILABLE"))
    overall_status = "PASS" if action_distribution_status == "PASS" and timing_and_bet_size_status == "PASS" else "FAIL"
    if timing_and_bet_size_status == "NOT_AVAILABLE" and action_distribution_status == "PASS":
        overall_status = "PARTIAL"
    return {
        "status": overall_status,
        "action_distribution_status": action_distribution_status,
        "timing_and_bet_size_status": timing_and_bet_size_status,
        "timing_bet_size": timing_bet_size,
        "examples": len(examples),
        "js_divergence": js,
        "max_js_divergence": max_js_divergence,
        "total_variation": tv,
        "max_total_variation": max_total_variation,
        "human_distribution": human_distribution,
        "model_distribution": model_distribution,
        "bucket_metrics": bucket_metrics,
        "weighted_bucket_js_divergence": weighted_bucket_js / bucket_weight if bucket_weight else None,
        "limitations": [
            "Action-distribution similarity is measured from reconstructed historical actions.",
            "Timing and bet-size likeness use deterministic proxies derived from frame deltas and stack movement until reviewed labels are available.",
        ],
    }


def unavailable_timing_bet_size_report() -> dict[str, Any]:
    return {
        "status": "NOT_AVAILABLE",
        "reason": "behavior_proxy_not_evaluated",
        "timing": {"status": "NOT_AVAILABLE"},
        "bet_size": {"status": "NOT_AVAILABLE"},
    }


def evaluate_timing_bet_size_likeness(
    model: Any,
    records: list[tuple[PredictionRequest, dict[str, float], str]],
    dataset_dir: Path,
    *,
    max_behavior_rows: int = 200000,
    max_model_examples: int = 2500,
    min_behavior_samples: int = 100,
    max_timing_js_divergence: float = 0.18,
    max_bet_size_js_divergence: float = 0.22,
) -> dict[str, Any]:
    human = extract_human_behavior_proxies(dataset_dir, max_rows=max_behavior_rows)
    model_proxy = extract_model_behavior_proxies(
        model,
        records[: max(0, max_model_examples)],
    )
    timing = compare_behavior_proxy(
        human["timing_counts"],
        model_proxy["timing_counts"],
        keys=TIMING_BUCKETS,
        min_human_samples=min_behavior_samples,
        min_model_samples=min_behavior_samples,
        max_js_divergence=max_timing_js_divergence,
        insufficient_model_status="FAIL",
    )
    bet_size = compare_behavior_proxy(
        human["bet_size_counts"],
        model_proxy["bet_size_counts"],
        keys=BET_SIZE_BUCKETS,
        min_human_samples=min_behavior_samples,
        min_model_samples=max(10, min_behavior_samples // 4),
        max_js_divergence=max_bet_size_js_divergence,
        insufficient_model_status="FAIL",
    )
    statuses = {timing["status"], bet_size["status"]}
    if "FAIL" in statuses:
        status = "FAIL"
    elif "NOT_AVAILABLE" in statuses:
        status = "NOT_AVAILABLE"
    else:
        status = "PASS"
    return {
        "status": status,
        "method": "frame_delta_and_stack_movement_proxy",
        "timing": timing,
        "bet_size": bet_size,
        "human_samples": human["samples"],
        "model_samples": model_proxy["samples"],
        "limits": {
            "max_behavior_rows": max_behavior_rows,
            "max_model_examples": max_model_examples,
            "min_behavior_samples": min_behavior_samples,
            "max_timing_js_divergence": max_timing_js_divergence,
            "max_bet_size_js_divergence": max_bet_size_js_divergence,
        },
        "limitations": [
            "Action delay is approximated from frame gaps between recorded actions.",
            "Bet size is approximated from positive stack decreases in stack_events.csv.",
            "These proxies are acceptance gates for engineering progress, not a substitute for reviewed timing and wager labels.",
        ],
    }


def extract_human_behavior_proxies(dataset_dir: Path, *, max_rows: int) -> dict[str, Any]:
    timing_counts: Counter[str] = Counter()
    bet_size_counts: Counter[str] = Counter()
    action_rows = 0
    stack_rows = 0
    last_frame_by_hand_street: dict[tuple[str, str], float] = {}
    last_stack_by_player: dict[tuple[str, str], float] = {}
    chip_actions_by_player: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    chip_action_frames_by_player: dict[tuple[str, str], list[float]] = {}

    actions_path = dataset_dir / "actions.csv"
    if actions_path.exists():
        with actions_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if action_rows >= max_rows:
                    break
                action_rows += 1
                raw_action = str(row.get("action") or "").strip().lower()
                if raw_action == "all_in":
                    raw_action = "raise"
                if raw_action not in ACTION_ORDER:
                    continue
                frame = _safe_float(row.get("frame_id"))
                if frame is None:
                    continue
                key = (str(row.get("hand_id") or ""), str(row.get("street") or ""))
                previous_frame = last_frame_by_hand_street.get(key)
                if previous_frame is not None and frame > previous_frame:
                    timing_counts[bucket_delay_frames(frame - previous_frame)] += 1
                last_frame_by_hand_street[key] = frame
                if raw_action in CHIP_COMMITMENT_ACTIONS:
                    player_key = (str(row.get("hand_id") or ""), str(row.get("player_position") or ""))
                    chip_actions_by_player[player_key].append((frame, normalize_action(raw_action)))

    for actions in chip_actions_by_player.values():
        actions.sort(key=lambda item: item[0])
    chip_action_frames_by_player = {
        key: [frame for frame, _action in actions]
        for key, actions in chip_actions_by_player.items()
    }

    stack_path = dataset_dir / "stack_events.csv"
    if stack_path.exists():
        with stack_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if stack_rows >= max_rows:
                    break
                stack_rows += 1
                amount = _stack_decrease_amount(row, last_stack_by_player)
                if (
                    amount is not None
                    and amount > 0.0
                    and _stack_event_matches_chip_action(
                        row,
                        chip_actions_by_player,
                        chip_action_frames_by_player,
                    )
                ):
                    bet_size_counts[bucket_bet_size(amount)] += 1

    return {
        "timing_counts": timing_counts,
        "bet_size_counts": bet_size_counts,
          "samples": {
              "action_rows_scanned": action_rows,
              "stack_rows_scanned": stack_rows,
              "timing_samples": sum(timing_counts.values()),
              "bet_size_samples": sum(bet_size_counts.values()),
              "bet_size_match_window_frames": STACK_ACTION_MATCH_WINDOW_FRAMES,
          },
      }


def extract_model_behavior_proxies(
    model: Any,
    records: list[tuple[PredictionRequest, dict[str, float], str]],
) -> dict[str, Any]:
    timing_counts: Counter[str] = Counter()
    bet_size_counts: Counter[str] = Counter()
    records_scored = 0
    positive_bet_sizes = 0
    for request, features, _label in records:
        action, probabilities = model.predict_from_features(features)
        confidence = max((float(value) for value in probabilities.values()), default=0.0)
        plan = build_action_plan(request, normalize_action(action), confidence=confidence)
        timing_counts[bucket_wait_time_ms(plan.wait_time_ms)] += 1
        records_scored += 1
        if plan.bet_size > 0.0:
            positive_bet_sizes += 1
            bet_size_counts[bucket_bet_size(plan.bet_size)] += 1
    return {
        "timing_counts": timing_counts,
        "bet_size_counts": bet_size_counts,
        "samples": {
            "records_scored": records_scored,
            "timing_samples": sum(timing_counts.values()),
            "positive_bet_size_samples": positive_bet_sizes,
        },
    }


def compare_behavior_proxy(
    human_counts: Counter[str],
    model_counts: Counter[str],
    *,
    keys: Iterable[str],
    min_human_samples: int,
    min_model_samples: int,
    max_js_divergence: float,
    insufficient_model_status: str,
) -> dict[str, Any]:
    key_tuple = tuple(keys)
    human_samples = sum(human_counts.values())
    model_samples = sum(model_counts.values())
    if human_samples < min_human_samples:
        return {
            "status": "NOT_AVAILABLE",
            "reason": "insufficient_human_proxy_samples",
            "human_samples": human_samples,
            "model_samples": model_samples,
            "min_human_samples": min_human_samples,
        }
    if model_samples < min_model_samples:
        return {
            "status": insufficient_model_status,
            "reason": "insufficient_model_proxy_samples",
            "human_samples": human_samples,
            "model_samples": model_samples,
            "min_model_samples": min_model_samples,
        }
    human_distribution = normalize_distribution_for_keys(human_counts, key_tuple)
    model_distribution = normalize_distribution_for_keys(model_counts, key_tuple)
    js = jensen_shannon_for_keys(human_distribution, model_distribution, key_tuple)
    tv = total_variation_for_keys(human_distribution, model_distribution, key_tuple)
    return {
        "status": "PASS" if js <= max_js_divergence else "FAIL",
        "human_samples": human_samples,
        "model_samples": model_samples,
        "js_divergence": js,
        "max_js_divergence": max_js_divergence,
        "total_variation": tv,
        "human_distribution": human_distribution,
        "model_distribution": model_distribution,
    }


def bucket_delay_frames(delta_frames: float) -> str:
    if delta_frames <= 2.0:
        return "instant"
    if delta_frames <= 10.0:
        return "short"
    if delta_frames <= 30.0:
        return "medium"
    return "long"


def bucket_wait_time_ms(wait_time_ms: float) -> str:
    estimated_frames = max(float(wait_time_ms), 0.0) / 33.333
    return bucket_delay_frames(estimated_frames)


def bucket_bet_size(amount: float) -> str:
    value = max(float(amount), 0.0)
    if value <= 1.0:
        return "micro"
    if value <= 4.0:
        return "small"
    if value <= 12.0:
        return "medium"
    return "large"


def _stack_event_matches_chip_action(
    row: dict[str, str],
    chip_actions_by_player: dict[tuple[str, str], list[tuple[float, str]]],
    chip_action_frames_by_player: dict[tuple[str, str], list[float]],
) -> bool:
    key = (str(row.get("hand_id") or ""), str(row.get("player_position") or ""))
    frame = _safe_float(row.get("frame_id"))
    actions = chip_actions_by_player.get(key)
    frames = chip_action_frames_by_player.get(key)
    if frame is None or not actions or not frames:
        return False
    index = bisect.bisect_right(frames, frame) - 1
    if index < 0:
        return False
    matched_frame, _matched_action = actions[index]
    return 0.0 <= frame - matched_frame <= STACK_ACTION_MATCH_WINDOW_FRAMES


def _stack_decrease_amount(
    row: dict[str, str],
    last_stack_by_player: dict[tuple[str, str], float],
) -> float | None:
    key = (str(row.get("hand_id") or ""), str(row.get("player_position") or ""))
    diff = _safe_float(row.get("diff"))
    stack_after_event = _safe_float(row.get("stack_after_event"))
    stack = stack_after_event if stack_after_event is not None else _safe_float(row.get("stack"))
    amount: float | None = None
    if diff is not None and diff < 0.0:
        amount = abs(diff)
    elif stack is not None:
        previous = last_stack_by_player.get(key)
        if previous is not None and stack < previous:
            amount = previous - stack
    if stack is not None:
        last_stack_by_player[key] = stack
    return amount


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def random_card_deck(rng: random.Random) -> list[str]:
    deck = [rank + suit for rank in RANKS for suit in SUITS]
    rng.shuffle(deck)
    return deck


def random_request(rng: random.Random, *, player_count: int) -> PredictionRequest:
    deck = random_card_deck(rng)
    hole_cards = [deck.pop(), deck.pop()]
    street = rng.choices(STREETS, weights=[0.46, 0.27, 0.17, 0.10], k=1)[0]
    board_count = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]
    board_cards = [deck.pop() for _ in range(board_count)]
    pot = round(rng.uniform(1.5, 28.0), 2)
    facing_bet = rng.random() < 0.58
    to_call = round(rng.uniform(0.5, min(12.0, pot * 0.75)), 2) if facing_bet else 0.0
    stack = round(rng.uniform(12.0, 180.0), 2)
    min_raise = round(max(1.0, to_call * 2.0, pot * rng.uniform(0.25, 0.75)), 2)
    history = _random_history(rng, street, player_count=player_count, facing_bet=facing_bet, to_call=to_call)
    return PredictionRequest(
        position=rng.choice(POSITIONS),
        street=street,
        hole_cards=hole_cards,
        board_cards=board_cards,
        pot=pot,
        to_call=to_call,
        stack=stack,
        min_raise=min_raise,
        player_count=player_count,
        betting_history=history,
    )


def _random_history(
    rng: random.Random,
    street: str,
    *,
    player_count: int,
    facing_bet: bool,
    to_call: float,
) -> list[dict[str, Any]]:
    action_count = rng.randint(0, min(5, player_count))
    actions: list[dict[str, Any]] = []
    for index in range(action_count):
        if facing_bet and index == action_count - 1:
            action = rng.choice(["bet", "raise"])
            amount = max(to_call, rng.uniform(1.0, 8.0))
        else:
            action = rng.choices(["fold", "call", "check", "bet", "raise"], weights=[0.24, 0.26, 0.24, 0.14, 0.12], k=1)[0]
            amount = rng.uniform(0.5, 8.0) if action in {"call", "bet", "raise"} else 0.0
        actions.append(
            {
                "position": POSITIONS[index % len(POSITIONS)],
                "action": action,
                "amount": round(amount, 2),
                "street": street,
            }
        )
    return actions


def action_ev(request: PredictionRequest, action: str, rng: random.Random) -> float:
    action = normalize_action(action)
    features = request_to_features(request)
    strength = float(features.get("strength_proxy", 0.0))
    pot = max(request.pot, 1.0)
    to_call = max(request.to_call, 0.0)
    stack = max(request.stack, 1.0)
    pressure = min(to_call / pot, 1.0)
    street_index = float(features.get("street_index", 0.0))
    win_probability = min(max(0.10 + 0.78 * strength - 0.10 * pressure + 0.02 * street_index, 0.05), 0.95)
    showdown_win = rng.random() < win_probability

    if action == "fold":
        return 0.0 if to_call > 0 else -0.03 * pot
    if action == "check":
        if to_call > 0:
            return -0.35 * to_call
        return pot * 0.12 if showdown_win else -pot * 0.04
    if action == "call":
        if to_call <= 0:
            return pot * 0.06 if showdown_win else -pot * 0.02
        return pot if showdown_win else -to_call

    if action == "bet":
        bet_size = min(max(request.min_raise, pot * 0.55), stack * 0.25)
        fold_equity = min(max(0.12 + 0.34 * strength - 0.05 * street_index, 0.05), 0.55)
        if rng.random() < fold_equity:
            return pot
        return pot + bet_size if showdown_win else -bet_size

    raise_size = min(max(request.min_raise, pot * 0.85), stack * 0.35)
    fold_equity = min(max(0.15 + 0.42 * strength - 0.18 * pressure - 0.04 * street_index, 0.04), 0.62)
    if rng.random() < fold_equity:
        return pot
    return pot + raise_size if showdown_win else -raise_size


def simulate_policy(
    model: Any,
    *,
    config: SimulationConfig,
) -> dict[str, Any]:
    rng = random.Random(config.seed)
    baseline = RuleBasedAgent()
    policy_rewards: list[float] = []
    baseline_rewards: list[float] = []
    policy_actions: Counter[str] = Counter()
    baseline_actions: Counter[str] = Counter()

    for _ in range(config.hands):
        request = random_request(rng, player_count=config.player_count)
        features = request_to_features(request)
        policy_action = prediction_action(model, features)
        baseline_response = baseline.predict(request)
        baseline_action = normalize_action(baseline_response.action)
        policy_actions[policy_action] += 1
        baseline_actions[baseline_action] += 1
        policy_rewards.append(action_ev(request, policy_action, rng))
        baseline_rewards.append(action_ev(request, baseline_action, rng))

    policy_wins = sum(1 for reward in policy_rewards if reward > 0)
    baseline_wins = sum(1 for reward in baseline_rewards if reward > 0)
    reward_deltas = [policy - baseline for policy, baseline in zip(policy_rewards, baseline_rewards)]
    return {
        "seed": config.seed,
        "hands": config.hands,
        "policy_win_rate": policy_wins / config.hands if config.hands else 0.0,
        "baseline_win_rate": baseline_wins / config.hands if config.hands else 0.0,
        "policy_mean_ev": sum(policy_rewards) / config.hands if config.hands else 0.0,
        "baseline_mean_ev": sum(baseline_rewards) / config.hands if config.hands else 0.0,
        "mean_ev_delta_vs_baseline": sum(reward_deltas) / config.hands if config.hands else 0.0,
        "policy_action_distribution": normalize_distribution(policy_actions),
        "baseline_action_distribution": normalize_distribution(baseline_actions),
    }


def aggregate_simulations(
    results: list[dict[str, Any]],
    *,
    min_win_rate_high_confidence: float = 0.505,
    min_win_rate_median: float = 0.52,
) -> dict[str, Any]:
    if not results:
        return {
            "status": "FAIL",
            "reason": "no_simulation_results",
            "runs": [],
        }
    win_rates = [float(row["policy_win_rate"]) for row in results]
    ev_deltas = [float(row["mean_ev_delta_vs_baseline"]) for row in results]
    probability_above_high_confidence = sum(1 for value in win_rates if value >= min_win_rate_high_confidence) / len(win_rates)
    probability_above_median = sum(1 for value in win_rates if value >= min_win_rate_median) / len(win_rates)
    status = (
        "PASS"
        if probability_above_high_confidence >= 0.90 and probability_above_median >= 0.50
        else "FAIL"
    )
    return {
        "status": status,
        "simulation_type": "synthetic_policy_proxy",
        "runs": results,
        "run_count": len(results),
        "mean_policy_win_rate": sum(win_rates) / len(win_rates),
        "min_policy_win_rate": min(win_rates),
        "max_policy_win_rate": max(win_rates),
        "mean_ev_delta_vs_baseline": sum(ev_deltas) / len(ev_deltas),
        "probability_win_rate_at_least_50_5": probability_above_high_confidence,
        "probability_win_rate_at_least_52": probability_above_median,
        "min_win_rate_high_confidence": min_win_rate_high_confidence,
        "min_win_rate_median": min_win_rate_median,
        "limitations": [
            "This is a synthetic policy proxy, not a full no-limit Hold'em self-play engine.",
            "It is useful as a regression gate and smoke simulation, but it cannot prove profitability.",
        ],
    }


def parse_seed_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).replace(";", ",").split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(text))
    return values or [42]


def run_simulations(
    model: Any,
    *,
    seeds: Iterable[int],
    hands_per_seed: int,
    player_count: int,
    min_win_rate_high_confidence: float,
    min_win_rate_median: float,
) -> dict[str, Any]:
    return run_holdem_self_play(
        model,
        seeds=list(seeds),
        hands_per_seed=hands_per_seed,
        player_count=player_count,
        min_win_rate_high_confidence=min_win_rate_high_confidence,
        min_win_rate_median=min_win_rate_median,
    )
