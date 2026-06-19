from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from poker_agent.agents import RuleBasedAgent
from poker_agent.features import request_to_features
from poker_agent.schemas import PredictionRequest


ACTION_ORDER = ("fold", "call", "check", "bet", "raise")
RANKS = "23456789TJQKA"
SUITS = "SHDC"
POSITIONS = ("SB", "BB", "UTG", "MP", "CO", "BTN")
STREETS = ("preflop", "flop", "turn", "river")


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


def expected_action_ev(features: dict[str, float], action: str) -> float:
    action = normalize_action(action)
    strength = float(features.get("strength_proxy", 0.0))
    pot = max(float(features.get("pot", 0.0)), 1.0)
    to_call = max(float(features.get("to_call", 0.0)), 0.0)
    stack = max(float(features.get("stack", 0.0)), 1.0)
    min_raise = max(float(features.get("min_raise", 0.0)), 1.0)
    pressure = min(to_call / pot, 1.0)
    street_index = float(features.get("street_index", 0.0))
    win_probability = min(max(0.10 + 0.78 * strength - 0.10 * pressure + 0.02 * street_index, 0.05), 0.95)

    if action == "fold":
        return 0.0 if to_call > 0 else -0.03 * pot
    if action == "check":
        if to_call > 0:
            return -10.0 * pot
        return win_probability * (0.12 * pot) - (1.0 - win_probability) * (0.04 * pot)
    if action == "call":
        if to_call <= 0:
            return win_probability * (0.06 * pot) - (1.0 - win_probability) * (0.02 * pot)
        return win_probability * pot - (1.0 - win_probability) * to_call
    if action == "bet":
        if to_call > 0:
            return -10.0 * pot
        bet_size = min(max(min_raise, pot * 0.55), stack * 0.25)
        fold_equity = min(max(0.12 + 0.34 * strength - 0.05 * street_index, 0.05), 0.55)
        called_ev = win_probability * (pot + bet_size) - (1.0 - win_probability) * bet_size
        return fold_equity * pot + (1.0 - fold_equity) * called_ev

    raise_size = min(max(min_raise, pot * 0.85), stack * 0.35)
    fold_equity = min(max(0.15 + 0.42 * strength - 0.18 * pressure - 0.04 * street_index, 0.04), 0.62)
    called_ev = win_probability * (pot + raise_size) - (1.0 - win_probability) * raise_size
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


class EVCalibratedPolicy:
    """Wrap a supervised policy with a deterministic EV-aware action selector."""

    def __init__(
        self,
        base_model: Any,
        *,
        ev_weight: float = 0.72,
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
    timing_and_bet_size_status = "NOT_AVAILABLE"
    return {
        "status": action_distribution_status,
        "action_distribution_status": action_distribution_status,
        "timing_and_bet_size_status": timing_and_bet_size_status,
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
            "Timing and bet-size likeness require reviewed delay and amount labels; those fields are not reliably present in actions.csv.",
        ],
    }


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
    results = [
        simulate_policy(
            model,
            config=SimulationConfig(hands=hands_per_seed, seed=seed, player_count=player_count),
        )
        for seed in seeds
    ]
    return aggregate_simulations(
        results,
        min_win_rate_high_confidence=min_win_rate_high_confidence,
        min_win_rate_median=min_win_rate_median,
    )
