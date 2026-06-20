from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from poker_agent.action_planning import build_action_plan
from poker_agent.agents import RuleBasedAgent
from poker_agent.features import request_to_features
from poker_agent.schemas import PredictionRequest


RANKS = "23456789TJQKA"
SUITS = "SHDC"
RANK_VALUE = {rank: index + 2 for index, rank in enumerate(RANKS)}
POSITIONS = ("BTN", "SB", "BB", "UTG", "MP", "CO")
STREETS = ("preflop", "flop", "turn", "river")
ACTION_ORDER = ("fold", "call", "check", "bet", "raise")
PAIRED_RESULT_EPSILON = 1e-9


@dataclass
class SeatState:
    index: int
    position: str
    hole_cards: list[str]
    stack: float
    committed: float = 0.0
    street_committed: float = 0.0
    active: bool = True
    all_in: bool = False


@dataclass
class HandResult:
    hero_delta: float
    hero_positive: bool
    hero_showdown_win: bool
    action_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class HoldemSelfPlayConfig:
    hands: int = 500
    seed: int = 42
    player_count: int = 6
    starting_stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0


def deck(rng: random.Random) -> list[str]:
    cards = [rank + suit for rank in RANKS for suit in SUITS]
    rng.shuffle(cards)
    return cards


def normalize_action(action: str) -> str:
    text = str(action or "").strip().lower()
    if text == "all_in":
        return "raise"
    if text in ACTION_ORDER:
        return text
    return "fold"


def evaluate_seven_cards(cards: list[str]) -> tuple[int, tuple[int, ...]]:
    if len(cards) < 5:
        raise ValueError("At least five cards are required for showdown evaluation")
    return max(evaluate_five_cards(list(combo)) for combo in combinations(cards, 5))


def evaluate_five_cards(cards: list[str]) -> tuple[int, tuple[int, ...]]:
    ranks = sorted((RANK_VALUE[card[0].upper()] for card in cards), reverse=True)
    suits = [card[1].upper() for card in cards]
    counts = Counter(ranks)
    grouped = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if flush and straight_high:
        return 8, (straight_high,)
    if grouped[0][1] == 4:
        quad = grouped[0][0]
        kicker = max(rank for rank in ranks if rank != quad)
        return 7, (quad, kicker)
    if grouped[0][1] == 3 and grouped[1][1] == 2:
        return 6, (grouped[0][0], grouped[1][0])
    if flush:
        return 5, tuple(ranks)
    if straight_high:
        return 4, (straight_high,)
    if grouped[0][1] == 3:
        trips = grouped[0][0]
        kickers = tuple(rank for rank in ranks if rank != trips)
        return 3, (trips, *kickers)
    pairs = [rank for rank, count in grouped if count == 2]
    if len(pairs) >= 2:
        high_pair, low_pair = sorted(pairs, reverse=True)[:2]
        kicker = max(rank for rank in ranks if rank not in {high_pair, low_pair})
        return 2, (high_pair, low_pair, kicker)
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = tuple(rank for rank in ranks if rank != pair)
        return 1, (pair, *kickers)
    return 0, tuple(ranks)


def _straight_high(ranks: list[int]) -> int:
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    for index in range(len(unique) - 4):
        window = unique[index : index + 5]
        if window[0] - window[-1] == 4 and len(window) == 5:
            return window[0]
    return 0


def simulate_holdem_hand(
    model: Any,
    *,
    seed: int,
    hero_uses_model: bool,
    config: HoldemSelfPlayConfig,
) -> HandResult:
    rng = random.Random(seed)
    cards = deck(rng)
    player_count = max(2, min(config.player_count, len(POSITIONS)))
    seats = [
        SeatState(
            index=index,
            position=POSITIONS[index],
            hole_cards=[cards.pop(), cards.pop()],
            stack=float(config.starting_stack),
        )
        for index in range(player_count)
    ]
    board = [cards.pop() for _ in range(5)]
    pot = 0.0
    history: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    baseline = RuleBasedAgent()

    pot += _post_blind(seats[1 % player_count], config.small_blind)
    pot += _post_blind(seats[2 % player_count], config.big_blind)

    for street in STREETS:
        for seat in seats:
            seat.street_committed = 0.0
        if street == "preflop":
            seats[1 % player_count].street_committed = min(config.small_blind, seats[1 % player_count].committed)
            seats[2 % player_count].street_committed = min(config.big_blind, seats[2 % player_count].committed)
            current_bet = config.big_blind
        else:
            current_bet = 0.0

        for index in _action_order(street, player_count):
            seat = seats[index]
            if not seat.active or seat.all_in:
                continue
            if _active_count(seats) <= 1:
                break

            to_call = max(0.0, current_bet - seat.street_committed)
            visible_board = _visible_board(board, street)
            request = PredictionRequest(
                position=seat.position,
                street=street,
                hole_cards=list(seat.hole_cards),
                board_cards=visible_board,
                pot=round(pot, 2),
                to_call=round(to_call, 2),
                stack=round(max(seat.stack, 0.0), 2),
                min_raise=round(max(config.big_blind, current_bet + config.big_blind), 2),
                player_count=player_count,
                betting_history=list(history),
            )
            action, confidence = _select_action(
                model,
                baseline,
                request,
                hero_uses_model=hero_uses_model and index == 0,
            )
            wager = _wager_for_action(request, action, confidence=confidence)
            if to_call > 0.0 and action in {"check", "bet"}:
                action = "call" if wager >= to_call else "fold"
            if to_call <= 0.0 and action == "call":
                action = "check"

            committed = _apply_action(seat, action, wager=wager, to_call=to_call)
            if committed > 0.0:
                pot += committed
                current_bet = max(current_bet, seat.street_committed)
            action_counts[action] += 1
            history.append(
                {
                    "position": seat.position,
                    "action": action,
                    "amount": round(committed, 2),
                    "street": street,
                }
            )

        if _active_count(seats) <= 1:
            break

    winners = _winners(seats, board)
    if winners:
        share = pot / len(winners)
        for winner in winners:
            winner.stack += share
    hero = seats[0]
    delta = hero.stack - config.starting_stack
    return HandResult(
        hero_delta=delta,
        hero_positive=delta > 0.0,
        hero_showdown_win=hero in winners and _active_count(seats) > 1,
        action_counts=action_counts,
    )


def run_holdem_self_play(
    model: Any,
    *,
    seeds: list[int],
    hands_per_seed: int,
    player_count: int,
    min_win_rate_high_confidence: float,
    min_win_rate_median: float,
) -> dict[str, Any]:
    runs = [
        _run_seed(
            model,
            seed=seed,
            hands=hands_per_seed,
            player_count=player_count,
        )
        for seed in seeds
    ]
    paired_rates = [float(row["policy_win_rate"]) for row in runs]
    ev_deltas = [float(row["mean_ev_delta_vs_baseline"]) for row in runs]
    probability_above_high_confidence = sum(
        1 for value in paired_rates if value >= min_win_rate_high_confidence
    ) / len(paired_rates)
    probability_above_median = sum(1 for value in paired_rates if value >= min_win_rate_median) / len(paired_rates)
    status = (
        "PASS"
        if probability_above_high_confidence >= 0.90 and probability_above_median >= 0.50
        else "FAIL"
    )
    return {
        "status": status,
        "simulation_type": "validated_multi_agent_holdem_self_play",
        "environment": {
            "name": "bounded_no_limit_holdem_self_play_v1",
            "showdown_evaluator": "complete_5_of_7_card_ranker",
            "streets": list(STREETS),
            "players": player_count,
            "comparison": "paired_policy_vs_rule_baseline_on_identical_deals",
        },
        "runs": runs,
        "run_count": len(runs),
        "mean_policy_win_rate": sum(paired_rates) / len(paired_rates) if paired_rates else 0.0,
        "min_policy_win_rate": min(paired_rates) if paired_rates else 0.0,
        "max_policy_win_rate": max(paired_rates) if paired_rates else 0.0,
        "mean_ev_delta_vs_baseline": sum(ev_deltas) / len(ev_deltas) if ev_deltas else 0.0,
        "probability_win_rate_at_least_50_5": probability_above_high_confidence,
        "probability_win_rate_at_least_52": probability_above_median,
        "min_win_rate_high_confidence": min_win_rate_high_confidence,
        "min_win_rate_median": min_win_rate_median,
        "policy_win_rate_definition": "paired_score_against_rule_baseline_win_1_tie_0_5_loss_0",
        "limitations": [
            "The simulator uses a bounded one-decision-per-player betting abstraction per street.",
            "The showdown evaluator ranks real Hold'em hands from private cards and board cards.",
            "This is a reproducible acceptance environment; profitability still requires larger opponent pools and reviewed betting abstractions.",
        ],
    }


def _run_seed(model: Any, *, seed: int, hands: int, player_count: int) -> dict[str, Any]:
    policy_deltas: list[float] = []
    baseline_deltas: list[float] = []
    policy_positive = 0
    baseline_positive = 0
    paired_wins = 0.0
    policy_actions: Counter[str] = Counter()
    baseline_actions: Counter[str] = Counter()
    config = HoldemSelfPlayConfig(hands=hands, seed=seed, player_count=player_count)
    for hand_index in range(hands):
        hand_seed = seed * 1_000_003 + hand_index
        policy_result = simulate_holdem_hand(
            model,
            seed=hand_seed,
            hero_uses_model=True,
            config=config,
        )
        baseline_result = simulate_holdem_hand(
            model,
            seed=hand_seed,
            hero_uses_model=False,
            config=config,
        )
        policy_deltas.append(policy_result.hero_delta)
        baseline_deltas.append(baseline_result.hero_delta)
        policy_positive += int(policy_result.hero_positive)
        baseline_positive += int(baseline_result.hero_positive)
        delta_gap = policy_result.hero_delta - baseline_result.hero_delta
        if delta_gap > PAIRED_RESULT_EPSILON:
            paired_wins += 1.0
        elif abs(delta_gap) <= PAIRED_RESULT_EPSILON:
            paired_wins += 0.5
        policy_actions.update(policy_result.action_counts)
        baseline_actions.update(baseline_result.action_counts)

    reward_deltas = [policy - baseline for policy, baseline in zip(policy_deltas, baseline_deltas)]
    return {
        "seed": seed,
        "hands": hands,
        "policy_win_rate": paired_wins / hands if hands else 0.0,
        "policy_positive_rate": policy_positive / hands if hands else 0.0,
        "baseline_positive_rate": baseline_positive / hands if hands else 0.0,
        "policy_mean_ev": sum(policy_deltas) / hands if hands else 0.0,
        "baseline_mean_ev": sum(baseline_deltas) / hands if hands else 0.0,
        "mean_ev_delta_vs_baseline": sum(reward_deltas) / hands if hands else 0.0,
        "policy_action_distribution": _normalize_counter(policy_actions),
        "baseline_action_distribution": _normalize_counter(baseline_actions),
    }


def _select_action(
    model: Any,
    baseline: RuleBasedAgent,
    request: PredictionRequest,
    *,
    hero_uses_model: bool,
) -> tuple[str, float]:
    if hero_uses_model:
        if hasattr(model, "predict_request"):
            action, probabilities = model.predict_request(request)
            return normalize_action(action), max((float(value) for value in probabilities.values()), default=0.0)
        if hasattr(model, "predict_from_features"):
            action, probabilities = model.predict_from_features(request_to_features(request))
            return normalize_action(action), max((float(value) for value in probabilities.values()), default=0.0)
        response = model.predict(request)
        return normalize_action(response.action), float(getattr(response, "confidence", 0.0))
    response = baseline.predict(request)
    return normalize_action(response.action), response.confidence


def _wager_for_action(request: PredictionRequest, action: str, *, confidence: float) -> float:
    plan = build_action_plan(request, action, confidence=confidence)
    if action == "call":
        return min(request.stack, request.to_call)
    if action in {"bet", "raise"}:
        return min(request.stack, max(plan.bet_size, request.to_call))
    return 0.0


def _apply_action(seat: SeatState, action: str, *, wager: float, to_call: float) -> float:
    action = normalize_action(action)
    if action == "fold":
        if to_call > 0.0:
            seat.active = False
        return 0.0
    if action in {"check", "call"}:
        amount = min(seat.stack, to_call)
    elif action in {"bet", "raise"}:
        amount = min(seat.stack, max(wager, to_call))
    else:
        amount = 0.0
    seat.stack -= amount
    seat.committed += amount
    seat.street_committed += amount
    seat.all_in = seat.stack <= 0.0 and seat.active
    return amount


def _post_blind(seat: SeatState, amount: float) -> float:
    committed = min(seat.stack, amount)
    seat.stack -= committed
    seat.committed += committed
    seat.street_committed += committed
    seat.all_in = seat.stack <= 0.0
    return committed


def _visible_board(board: list[str], street: str) -> list[str]:
    if street == "flop":
        return board[:3]
    if street == "turn":
        return board[:4]
    if street == "river":
        return board[:5]
    return []


def _action_order(street: str, player_count: int) -> list[int]:
    if street == "preflop":
        order = [3, 4, 5, 0, 1, 2]
    else:
        order = [1, 2, 3, 4, 5, 0]
    return [index for index in order if index < player_count]


def _active_count(seats: list[SeatState]) -> int:
    return sum(1 for seat in seats if seat.active)


def _winners(seats: list[SeatState], board: list[str]) -> list[SeatState]:
    active = [seat for seat in seats if seat.active]
    if len(active) <= 1:
        return active
    scores = {seat.index: evaluate_seven_cards(seat.hole_cards + board) for seat in active}
    best = max(scores.values())
    return [seat for seat in active if scores[seat.index] == best]


def _normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {action: counter.get(action, 0) / total for action in ACTION_ORDER}
