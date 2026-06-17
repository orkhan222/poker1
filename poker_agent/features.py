from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from poker_agent.schemas import PredictionRequest
from poker_agent.schemas import VALID_ACTIONS


RANK_TO_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
VISIBLE_BOARD_COUNTS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
ACTION_NORMALIZATION = {
    "all-in": "all_in",
    "all in": "all_in",
    "allin": "all_in",
    "calls": "call",
    "bets": "bet",
    "raises": "raise",
    "folds": "fold",
    "checks": "check",
}
NON_DECISION_ACTIONS = {
    "ante",
    "post_sb",
    "post_bb",
    "joined",
    "sit_out",
    "won",
    "muck",
}


def normalize_action(action: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(action).strip().lower())
    cleaned = ACTION_NORMALIZATION.get(cleaned, cleaned)
    if "fold" in cleaned:
        return "fold"
    if "check" in cleaned:
        return "check"
    if "call" in cleaned:
        return "call"
    if "raise" in cleaned:
        return "raise"
    if "bet" in cleaned:
        return "bet"
    if "all" in cleaned:
        return "all_in"
    return cleaned or "unknown"


def parse_cards(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(card).strip() for card in raw if str(card).strip()]
    return [part for part in str(raw).replace(",", " ").split() if part]


def visible_board_cards(board_cards: Iterable[str], street: str) -> list[str]:
    """Return only cards visible at the decision street to avoid future-card leakage."""
    visible_count = VISIBLE_BOARD_COUNTS.get(str(street or "preflop").lower(), 0)
    return list(board_cards)[:visible_count]


def safe_float(raw: Any, default: float = 0.0) -> float:
    if raw is None or raw == "":
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", ".")
    text = re.sub(r"[^0-9.+-]", "", text)
    if text.count(".") > 1:
        head, *tail = text.split(".")
        text = head + "." + "".join(tail)
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def card_rank(card: str) -> int:
    if not card:
        return 0
    return RANK_TO_VALUE.get(card[0].upper(), 0)


def card_suit(card: str) -> str:
    return card[1:].lower() if len(card) > 1 else ""


def hand_strength_proxy(hole_cards: Iterable[str]) -> float:
    cards = list(hole_cards)[:2]
    if len(cards) < 2:
        return 0.0
    ranks = [card_rank(card) for card in cards]
    suits = [card_suit(card) for card in cards]
    high = max(ranks) / 14.0
    low = min(ranks) / 14.0
    pair_bonus = 0.28 if ranks[0] == ranks[1] else 0.0
    suited_bonus = 0.08 if suits[0] and suits[0] == suits[1] else 0.0
    connector_bonus = 0.05 if abs(ranks[0] - ranks[1]) <= 1 else 0.0
    return min(1.0, 0.55 * high + 0.25 * low + pair_bonus + suited_bonus + connector_bonus)


def has_straight(ranks: Iterable[int]) -> bool:
    values = {rank for rank in ranks if rank > 0}
    if 14 in values:
        values.add(1)
    for start in range(1, 11):
        if all(rank in values for rank in range(start, start + 5)):
            return True
    return False


def straight_draw_score(ranks: Iterable[int]) -> float:
    values = {rank for rank in ranks if rank > 0}
    if 14 in values:
        values.add(1)
    best_window = 0
    for start in range(1, 11):
        best_window = max(best_window, sum(1 for rank in range(start, start + 5) if rank in values))
    if best_window >= 5:
        return 1.0
    if best_window == 4:
        return 0.75
    if best_window == 3:
        return 0.35
    return 0.0


def made_hand_category(cards: Iterable[str]) -> tuple[str, float]:
    ranks = [card_rank(card) for card in cards if card_rank(card)]
    suits = [card_suit(card) for card in cards if card_suit(card)]
    rank_counts = sorted((ranks.count(rank) for rank in set(ranks)), reverse=True)
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}
    flush = any(count >= 5 for count in suit_counts.values())
    straight = has_straight(ranks)
    if flush and straight:
        return "straight_flush", 1.0
    if rank_counts and rank_counts[0] == 4:
        return "quads", 0.88
    if len(rank_counts) >= 2 and rank_counts[0] == 3 and rank_counts[1] >= 2:
        return "full_house", 0.78
    if flush:
        return "flush", 0.68
    if straight:
        return "straight", 0.58
    if rank_counts and rank_counts[0] == 3:
        return "trips", 0.46
    if len([count for count in rank_counts if count >= 2]) >= 2:
        return "two_pair", 0.34
    if rank_counts and rank_counts[0] == 2:
        return "pair", 0.22
    return "high_card", 0.0


def preflop_bucket_features(hole_ranks: list[int], hole_suits: list[str]) -> dict[str, float]:
    if len(hole_ranks) < 2:
        return {
            "preflop_bucket_score": 0.0,
            "premium_pair": 0.0,
            "broadway_count": 0.0,
            "ace_high": 0.0,
        }

    high = max(hole_ranks)
    low = min(hole_ranks)
    pair = hole_ranks[0] == hole_ranks[1]
    suited = len(hole_suits) >= 2 and hole_suits[0] == hole_suits[1]
    gap = abs(hole_ranks[0] - hole_ranks[1])
    broadway_count = sum(1 for rank in hole_ranks if rank >= 10)

    score = 0.28 * (high / 14.0) + 0.18 * (low / 14.0)
    score += 0.26 if pair else 0.0
    score += 0.09 if suited else 0.0
    score += 0.08 if gap <= 1 else 0.0
    score += 0.08 if broadway_count == 2 else 0.0
    score += 0.05 if high == 14 else 0.0

    return {
        "preflop_bucket_score": min(score, 1.0),
        "premium_pair": 1.0 if pair and high >= 11 else 0.0,
        "broadway_count": broadway_count / 2.0,
        "ace_high": 1.0 if high == 14 else 0.0,
    }


def card_texture_features(hole_cards: Iterable[str], board_cards: Iterable[str]) -> dict[str, float]:
    hole = list(hole_cards)[:2]
    board = list(board_cards)
    hole_ranks = [card_rank(card) for card in hole]
    hole_suits = [card_suit(card) for card in hole if card_suit(card)]
    board_ranks = [card_rank(card) for card in board]
    board_suits = [card_suit(card) for card in board if card_suit(card)]
    all_cards = hole + board
    all_ranks = [card_rank(card) for card in all_cards]
    all_suits = [card_suit(card) for card in all_cards if card_suit(card)]

    high_rank = max(hole_ranks, default=0)
    low_rank = min(hole_ranks, default=0)
    rank_gap = abs(hole_ranks[0] - hole_ranks[1]) if len(hole_ranks) >= 2 else 0
    board_rank_counts = {rank: board_ranks.count(rank) for rank in set(board_ranks)}
    board_suit_counts = {suit: board_suits.count(suit) for suit in set(board_suits)}
    all_suit_counts = {suit: all_suits.count(suit) for suit in set(all_suits)}
    made_category, made_score = made_hand_category(all_cards)
    board_connectedness = straight_draw_score(board_ranks)
    combined_straight_draw = straight_draw_score(all_ranks)
    flush_draw_pressure = min(max(all_suit_counts.values(), default=0) / 5.0, 1.0)
    board_flush_pressure = min(max(board_suit_counts.values(), default=0) / 5.0, 1.0)
    board_wetness = min(1.0, 0.45 * board_connectedness + 0.45 * board_flush_pressure + 0.10 * (1.0 if any(count >= 2 for count in board_rank_counts.values()) else 0.0))
    board_high_rank = max(board_ranks, default=0)
    hole_overpair = len(hole_ranks) >= 2 and hole_ranks[0] == hole_ranks[1] and board_high_rank > 0 and hole_ranks[0] > board_high_rank
    top_pair_or_better = board_high_rank > 0 and made_score >= 0.22 and any(rank == board_high_rank for rank in hole_ranks)

    features = {
        "hole_high_rank": high_rank / 14.0,
        "hole_low_rank": low_rank / 14.0,
        "hole_pair": 1.0 if len(hole_ranks) >= 2 and hole_ranks[0] == hole_ranks[1] else 0.0,
        "hole_suited": 1.0 if len(hole_suits) >= 2 and hole_suits[0] == hole_suits[1] else 0.0,
        "hole_connected": 1.0 if len(hole_ranks) >= 2 and rank_gap <= 1 else 0.0,
        "hole_gap": min(rank_gap / 12.0, 1.0),
        "board_high_rank": board_high_rank / 14.0,
        "board_pair": 1.0 if any(count >= 2 for count in board_rank_counts.values()) else 0.0,
        "board_trips": 1.0 if any(count >= 3 for count in board_rank_counts.values()) else 0.0,
        "board_suited_pressure": board_flush_pressure,
        "board_connectedness": board_connectedness,
        "board_wetness": board_wetness,
        "made_hand_score": made_score,
        "straight_draw_score": combined_straight_draw,
        "flush_draw_pressure": flush_draw_pressure,
        "hole_overpair": 1.0 if hole_overpair else 0.0,
        "top_pair_or_better": 1.0 if top_pair_or_better else 0.0,
        f"made_hand={made_category}": 1.0,
    }
    features.update(preflop_bucket_features(hole_ranks, hole_suits))
    return features


def normalize_position_group(position: str) -> str:
    text = str(position or "").lower()
    if "bottom" in text:
        return "bottom"
    if "top" in text:
        return "top"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if text in {"btn", "button", "co", "mp", "utg", "sb", "bb"}:
        return text
    return "unknown"


def betting_history_to_features(history: Iterable[dict[str, Any]], hero_position: str) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    aggressive_count = 0
    last_action = "none"
    last_aggressor_group = "none"
    hero_has_acted = 0.0

    for raw_event in history:
        action = normalize_action(str(raw_event.get("action", "")))
        position = str(raw_event.get("position") or raw_event.get("player_position") or "")
        if action not in VALID_ACTIONS:
            continue
        counts[action] += 1
        last_action = action
        if position and position == hero_position:
            hero_has_acted = 1.0
        if action in {"bet", "raise", "all_in"}:
            aggressive_count += 1
            last_aggressor_group = normalize_position_group(position)

    total = sum(counts.values())
    features = {
        "hist_action_count": float(total),
        "hist_aggressive_count": float(aggressive_count),
        "hist_call_count": float(counts.get("call", 0)),
        "hist_check_count": float(counts.get("check", 0)),
        "hist_fold_count": float(counts.get("fold", 0)),
        "hist_raise_count": float(counts.get("raise", 0) + counts.get("all_in", 0)),
        "hist_aggression_ratio": aggressive_count / total if total else 0.0,
        "hist_hero_has_acted": hero_has_acted,
        f"hist_last_action={last_action}": 1.0,
        f"hist_last_aggressor_group={last_aggressor_group}": 1.0,
    }
    return features


def betting_context_features(
    *,
    action_count: int,
    aggressive_count: int,
    call_count: int,
    check_count: int,
    fold_count: int,
    players_acted: set[str],
    player_count: int,
    hero_commit: float,
    highest_commit: float,
    running_pot: float,
    stack: float,
    to_call: float,
    min_raise: float,
    last_aggressor_position: str,
    hero_position: str,
) -> dict[str, float]:
    pot_base = max(running_pot, highest_commit, 1.0)
    stack_base = max(stack + hero_commit, 1.0)
    raise_pressure = min(min_raise / max(stack, 1.0), 1.0) if stack > 0 else 0.0
    last_aggressor_group = normalize_position_group(last_aggressor_position) if last_aggressor_position else "none"

    return {
        "street_action_count": float(action_count),
        "street_aggressive_count": float(aggressive_count),
        "street_call_count": float(call_count),
        "street_check_count": float(check_count),
        "street_fold_count": float(fold_count),
        "street_aggression_ratio": aggressive_count / action_count if action_count else 0.0,
        "players_acted_ratio": len(players_acted) / max(player_count, 1),
        "hero_commitment_ratio": min(hero_commit / stack_base, 1.0),
        "table_commitment_pressure": min(highest_commit / pot_base, 1.0),
        "facing_bet_or_raise": 1.0 if to_call > 0 else 0.0,
        "call_price_ratio": min(to_call / max(stack, 1.0), 1.0) if stack > 0 else 0.0,
        "raise_pressure": raise_pressure,
        "last_aggressor_is_hero": 1.0 if last_aggressor_position and last_aggressor_position == hero_position else 0.0,
        f"last_aggressor_group={last_aggressor_group}": 1.0,
    }


def request_to_features(request: PredictionRequest) -> dict[str, float]:
    pot_odds = request.to_call / (request.pot + request.to_call) if request.pot + request.to_call > 0 else 0.0
    stack_to_pot = request.stack / request.pot if request.pot > 0 else 0.0
    spr = request.stack / (request.pot + request.to_call) if request.pot + request.to_call > 0 else 0.0
    raise_to_stack = request.min_raise / request.stack if request.stack > 0 else 0.0
    call_to_stack = request.to_call / request.stack if request.stack > 0 else 0.0
    position_group = normalize_position_group(request.position)
    features = {
        "bias": 1.0,
        "street_index": float(STREET_ORDER.get(request.street, 0)),
        "board_count": float(len(request.board_cards)),
        "hole_count": float(len(request.hole_cards)),
        "hole_cards_missing": 1.0 if len(request.hole_cards) < 2 else 0.0,
        "hole_card_observed_ratio": min(len(request.hole_cards) / 2.0, 1.0),
        "board_card_observed_ratio": (
            len(request.board_cards) / max(VISIBLE_BOARD_COUNTS.get(request.street, 0), 1)
            if request.street != "preflop"
            else 1.0
        ),
        "strength_proxy": hand_strength_proxy(request.hole_cards),
        "pot": request.pot,
        "to_call": request.to_call,
        "stack": request.stack,
        "min_raise": request.min_raise,
        "pot_odds": pot_odds,
        "call_to_stack": min(call_to_stack, 1.0),
        "raise_to_stack": min(raise_to_stack, 1.0),
        "has_call": 1.0 if request.to_call > 0 else 0.0,
        "stack_to_pot": min(stack_to_pot, 100.0),
        "spr": min(spr, 100.0),
        "player_count": float(request.player_count),
        "is_hero_like_position": 1.0 if position_group == "bottom" else 0.0,
        f"position_group={position_group}": 1.0,
        f"street={request.street}": 1.0,
    }
    features.update(card_texture_features(request.hole_cards, request.board_cards))
    features.update(betting_history_to_features(request.betting_history, request.position))
    return features


PRIVATE_CARD_FEATURE_PREFIXES = (
    "hole_",
    "made_hand=",
)
PRIVATE_CARD_FEATURES = {
    "hole_count",
    "hole_cards_missing",
    "hole_card_observed_ratio",
    "strength_proxy",
    "preflop_bucket_score",
    "premium_pair",
    "broadway_count",
    "ace_high",
    "made_hand_score",
    "straight_draw_score",
    "flush_draw_pressure",
    "hole_overpair",
    "top_pair_or_better",
}


def public_context_features(features: dict[str, float]) -> dict[str, float]:
    """Remove private-card and card-combo signals for no-hole-card models."""
    return {
        name: value
        for name, value in features.items()
        if name not in PRIVATE_CARD_FEATURES
        and not any(name.startswith(prefix) for prefix in PRIVATE_CARD_FEATURE_PREFIXES)
    }


def load_stack_contributions(
    dataset_dir: Path,
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    stack_path = dataset_dir / "stack_events.csv"
    contributions: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    if not stack_path.exists():
        return contributions

    with stack_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            diff = safe_float(row.get("diff"))
            if diff >= 0:
                continue
            hand_id = row.get("hand_id", "")
            position = row.get("player_position", "")
            if not hand_id or not position:
                continue
            contributions[(hand_id, position)].append((safe_int(row.get("frame_id")), abs(diff)))

    for rows in contributions.values():
        rows.sort(key=lambda item: item[0])
    return contributions


def amount_near_frame(
    contributions: dict[tuple[str, str], list[tuple[int, float]]],
    used_events: set[tuple[str, str, int, float]],
    hand_id: str,
    position: str,
    frame_id: int,
    window: int = 45,
) -> float:
    best: tuple[int, float] | None = None
    best_distance: int | None = None
    for event_frame, amount in contributions.get((hand_id, position), []):
        key = (hand_id, position, event_frame, amount)
        if key in used_events:
            continue
        distance = abs(event_frame - frame_id)
        if distance > window:
            continue
        if best_distance is None or distance < best_distance:
            best = (event_frame, amount)
            best_distance = distance
    if best is None:
        return 0.0
    used_events.add((hand_id, position, best[0], best[1]))
    return best[1]


def iter_actions_by_hand(actions_path: Path) -> Iterable[tuple[str, list[dict[str, str]]]]:
    current_hand_id = ""
    rows: list[dict[str, str]] = []
    with actions_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hand_id = row.get("hand_id", "")
            if current_hand_id and hand_id != current_hand_id:
                yield current_hand_id, rows
                rows = []
            current_hand_id = hand_id
            rows.append(row)
    if current_hand_id:
        yield current_hand_id, rows


def estimate_big_blind(rows: list[dict[str, str]], contributions: dict[tuple[str, str], list[tuple[int, float]]]) -> float:
    for row in rows:
        if normalize_action(row.get("action", "")) != "post_bb":
            continue
        amount = amount_near_frame(contributions, set(), row.get("hand_id", ""), row.get("player_position", ""), safe_int(row.get("frame_id")))
        if amount > 0:
            return amount
    return 0.0


def load_training_examples(
    dataset_dir: Path,
    max_examples: int = 0,
    require_hole_cards: bool = True,
    merge_all_in: bool = True,
    missing_hole_cards: str | None = None,
    include_hand_id: bool = False,
    include_request: bool = False,
) -> list[Any]:
    actions_path = dataset_dir / "actions.csv"
    players_path = dataset_dir / "players.csv"
    hands_path = dataset_dir / "hands.csv"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions.csv: {actions_path}")

    players_by_hand_pos: dict[tuple[str, str], dict[str, str]] = {}
    player_counts_by_hand: dict[str, int] = defaultdict(int)
    if players_path.exists():
        with players_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                hand_id = row.get("hand_id", "")
                position = row.get("position", "")
                players_by_hand_pos[(hand_id, position)] = row
                if safe_float(row.get("starting_stack")) > 0 or safe_float(row.get("ending_stack")) > 0:
                    player_counts_by_hand[hand_id] += 1

    hands_by_id: dict[str, dict[str, str]] = {}
    if hands_path.exists():
        with hands_path.open("r", newline="", encoding="utf-8") as handle:
            hands_by_id = {row.get("hand_id", ""): row for row in csv.DictReader(handle)}

    stack_contributions = load_stack_contributions(dataset_dir)
    missing_mode = missing_hole_cards or ("drop" if require_hole_cards else "flag")
    if missing_mode not in {"drop", "flag", "keep"}:
        raise ValueError("missing_hole_cards must be one of: drop, flag, keep")

    examples: list[Any] = []
    for hand_id, action_rows in iter_actions_by_hand(actions_path):
        hand = hands_by_id.get(hand_id, {})
        final_board_cards = parse_cards(hand.get("board_cards"))
        used_events: set[tuple[str, str, int, float]] = set()
        committed_by_street: dict[str, float] = defaultdict(float)
        current_street = ""
        last_raise_size = estimate_big_blind(action_rows, stack_contributions)
        big_blind = last_raise_size
        running_pot = 0.0
        street_action_count = 0
        street_aggressive_count = 0
        street_call_count = 0
        street_check_count = 0
        street_fold_count = 0
        players_acted: set[str] = set()
        last_aggressor_position = ""
        street_history_events: list[dict[str, Any]] = []

        for row in sorted(action_rows, key=lambda item: safe_int(item.get("frame_id"))):
            action = normalize_action(row.get("action", ""))
            if merge_all_in and action == "all_in":
                action = "raise"
            position = row.get("player_position", "")
            street = row.get("street", "preflop")
            frame_id = safe_int(row.get("frame_id"))
            if street != current_street:
                committed_by_street = defaultdict(float)
                current_street = street
                last_raise_size = big_blind
                street_action_count = 0
                street_aggressive_count = 0
                street_call_count = 0
                street_check_count = 0
                street_fold_count = 0
                players_acted = set()
                last_aggressor_position = ""
                street_history_events = []

            amount = amount_near_frame(stack_contributions, used_events, hand_id, position, frame_id)
            highest_commit = max(committed_by_street.values(), default=0.0)
            player_commit = committed_by_street[position]
            to_call = max(0.0, highest_commit - player_commit)
            min_raise = max(last_raise_size, big_blind, 0.0)

            if action in VALID_ACTIONS:
                player = players_by_hand_pos.get((hand_id, position), {})
                hole_cards = parse_cards(player.get("cards"))
                hole_cards_missing = len(hole_cards) < 2
                if missing_mode != "drop" or not hole_cards_missing:
                    pot = max(running_pot, sum(committed_by_street.values()), big_blind, 0.0)
                    stack = safe_float(player.get("starting_stack"))
                    if stack <= 0:
                        stack = safe_float(player.get("ending_stack"))
                    effective_stack = max(stack - player_commit, 0.0) if stack > 0 else 0.0

                    request = PredictionRequest(
                        position=position or "UNK",
                        street=street,
                        hole_cards=hole_cards,
                        board_cards=visible_board_cards(final_board_cards, street),
                        pot=pot,
                        to_call=to_call,
                        stack=effective_stack or stack,
                        min_raise=min_raise,
                        player_count=player_counts_by_hand.get(hand_id, 6) or 6,
                        betting_history=list(street_history_events),
                    )
                    features = request_to_features(request)
                    features.update(
                        betting_context_features(
                            action_count=street_action_count,
                            aggressive_count=street_aggressive_count,
                            call_count=street_call_count,
                            check_count=street_check_count,
                            fold_count=street_fold_count,
                            players_acted=players_acted,
                            player_count=player_counts_by_hand.get(hand_id, 6) or 6,
                            hero_commit=player_commit,
                            highest_commit=highest_commit,
                            running_pot=running_pot,
                            stack=effective_stack or stack,
                            to_call=to_call,
                            min_raise=min_raise,
                            last_aggressor_position=last_aggressor_position,
                            hero_position=position,
                        )
                    )
                    features["hole_cards_missing"] = 1.0 if hole_cards_missing else 0.0
                    features["hole_card_observed_ratio"] = min(len(hole_cards) / 2.0, 1.0)
                    features["board_card_observed_ratio"] = len(request.board_cards) / max(VISIBLE_BOARD_COUNTS.get(street, 0), 1) if street != "preflop" else 1.0
                    if include_request and include_hand_id:
                        examples.append((request, features, action, hand_id))
                    elif include_request:
                        examples.append((request, features, action))
                    elif include_hand_id:
                        examples.append((features, action, hand_id))
                    else:
                        examples.append((features, action))
                    if max_examples and len(examples) >= max_examples:
                        return examples

            if amount > 0 and action not in NON_DECISION_ACTIONS:
                before_highest = max(committed_by_street.values(), default=0.0)
                committed_by_street[position] += amount
                running_pot += amount
                after_commit = committed_by_street[position]
                if after_commit > before_highest:
                    last_raise_size = max(after_commit - before_highest, big_blind)
            elif amount > 0:
                committed_by_street[position] += amount
                running_pot += amount

            if action in VALID_ACTIONS:
                street_action_count += 1
                players_acted.add(position)
                if action in {"bet", "raise", "all_in"}:
                    street_aggressive_count += 1
                    last_aggressor_position = position
                elif action == "call":
                    street_call_count += 1
                elif action == "check":
                    street_check_count += 1
                elif action == "fold":
                    street_fold_count += 1
                street_history_events.append(
                    {
                        "position": position,
                        "action": action,
                        "amount": amount,
                        "frame_id": frame_id,
                        "street": street,
                    }
                )
    return examples
