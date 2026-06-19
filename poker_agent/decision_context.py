from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from poker_agent.features import request_to_features
from poker_agent.schemas import PredictionRequest


DECISION_OUTPUT_SCHEMA = {
    "action": "fold | call | check | bet | raise",
    "bet_size": "non-negative float; 0 for fold/check; call price for call; selected wager for bet/raise",
    "wait_time_ms": "integer; must be at least the model processing time",
    "confidence": "float between 0 and 1",
    "rationale": "short reason based only on the provided state",
}


POKER_RULES = (
    "Game type: Texas Hold'em poker.",
    "A player may fold only when continuing is optional or when facing a bet.",
    "A player may check only when to_call is zero.",
    "A player may call only when to_call is greater than zero.",
    "A player may bet only when no bet is currently faced on the street.",
    "A player may raise only when facing a bet or when the state indicates an aggressive betting opportunity.",
    "Preflop decisions depend strongly on hole-card strength, position, pot odds, stack depth, and prior aggression.",
    "Postflop decisions depend on visible board texture, made-hand strength, draw potential, pot odds, stack-to-pot ratio, and betting history.",
    "Do not infer hidden opponent cards or future board cards.",
    "When hole cards are missing, reduce confidence and prefer conservative actions unless the public context strongly supports aggression.",
)


FORMAL_GUIDELINES = (
    "Use only the structured state supplied in this prompt.",
    "Return exactly one supported action.",
    "Do not return all_in; map all-in pressure to raise.",
    "Do not invent cards, players, stack sizes, or previous actions.",
    "If an action is illegal in the supplied state, assign it low probability or do not select it.",
    "Prefer decisions that are consistent with pot odds and stack-to-pot ratio.",
    "For fold and check, bet_size must be 0.",
    "For call, bet_size should equal the current to_call amount.",
    "For bet or raise, bet_size must respect stack and min_raise constraints.",
    "wait_time_ms must be no lower than the inference processing time and should increase for complex or low-confidence states.",
    "Return compact JSON only. Do not include prose outside JSON.",
)


FEW_SHOT_EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "state": {
            "position": "BTN",
            "street": "preflop",
            "hole_cards": ["AS", "KD"],
            "board_cards": [],
            "pot": 2.5,
            "to_call": 1.0,
            "stack": 100.0,
            "min_raise": 4.5,
            "player_count": 6,
            "betting_history": [{"position": "UTG", "action": "raise", "amount": 3.0, "street": "preflop"}],
            "derived": {"pot_odds": 0.285714, "spr": 28.571429, "strength_proxy": 0.842857, "hole_cards_missing": False},
        },
        "output": {"action": "raise", "confidence": 0.72, "rationale": "premium hand with position and enough stack depth"},
    },
    {
        "state": {
            "position": "BB",
            "street": "preflop",
            "hole_cards": ["7C", "2D"],
            "board_cards": [],
            "pot": 3.0,
            "to_call": 5.0,
            "stack": 45.0,
            "min_raise": 10.0,
            "player_count": 6,
            "betting_history": [{"position": "CO", "action": "raise", "amount": 5.0, "street": "preflop"}],
            "derived": {"pot_odds": 0.625, "spr": 5.625, "strength_proxy": 0.353571, "hole_cards_missing": False},
        },
        "output": {"action": "fold", "confidence": 0.78, "rationale": "weak hand facing a large price"},
    },
    {
        "state": {
            "position": "CO",
            "street": "flop",
            "hole_cards": ["AH", "QH"],
            "board_cards": ["KH", "TH", "2C"],
            "pot": 12.0,
            "to_call": 0.0,
            "stack": 80.0,
            "min_raise": 6.0,
            "player_count": 5,
            "betting_history": [{"position": "SB", "action": "check", "amount": 0.0, "street": "flop"}],
            "derived": {"pot_odds": 0.0, "spr": 6.666667, "strength_proxy": 0.807143, "hole_cards_missing": False},
        },
        "output": {"action": "bet", "confidence": 0.67, "rationale": "strong draw and initiative opportunity when checked to"},
    },
    {
        "state": {
            "position": "MP",
            "street": "turn",
            "hole_cards": ["9S", "9D"],
            "board_cards": ["AC", "KD", "7H", "2S"],
            "pot": 20.0,
            "to_call": 14.0,
            "stack": 60.0,
            "min_raise": 28.0,
            "player_count": 4,
            "betting_history": [{"position": "UTG", "action": "bet", "amount": 14.0, "street": "turn"}],
            "derived": {"pot_odds": 0.411765, "spr": 1.764706, "strength_proxy": 0.632143, "hole_cards_missing": False},
        },
        "output": {"action": "fold", "confidence": 0.61, "rationale": "medium pair facing high pressure on overcard board"},
    },
    {
        "state": {
            "position": "BTN",
            "street": "river",
            "hole_cards": [],
            "board_cards": ["AS", "QS", "7D", "3C", "2H"],
            "pot": 18.0,
            "to_call": 0.0,
            "stack": 75.0,
            "min_raise": 9.0,
            "player_count": 3,
            "betting_history": [{"position": "SB", "action": "check", "amount": 0.0, "street": "river"}],
            "derived": {"pot_odds": 0.0, "spr": 4.166667, "strength_proxy": 0.0, "hole_cards_missing": True},
        },
        "output": {"action": "check", "confidence": 0.54, "rationale": "private cards are missing and no bet is faced"},
    },
    {
        "state": {
            "position": "SB",
            "street": "preflop",
            "hole_cards": ["JC", "JS"],
            "board_cards": [],
            "pot": 3.0,
            "to_call": 1.0,
            "stack": 90.0,
            "min_raise": 5.0,
            "player_count": 6,
            "betting_history": [{"position": "BTN", "action": "raise", "amount": 3.0, "street": "preflop"}],
            "derived": {"pot_odds": 0.25, "spr": 22.5, "strength_proxy": 0.832143, "hole_cards_missing": False},
        },
        "output": {"action": "raise", "confidence": 0.66, "rationale": "premium pair can apply pressure before the flop"},
    },
    {
        "state": {
            "position": "BB",
            "street": "flop",
            "hole_cards": ["8C", "8D"],
            "board_cards": ["8H", "4S", "2D"],
            "pot": 10.0,
            "to_call": 4.0,
            "stack": 70.0,
            "min_raise": 12.0,
            "player_count": 4,
            "betting_history": [{"position": "CO", "action": "bet", "amount": 4.0, "street": "flop"}],
            "derived": {"pot_odds": 0.285714, "spr": 5.0, "strength_proxy": 0.685714, "hole_cards_missing": False},
        },
        "output": {"action": "raise", "confidence": 0.74, "rationale": "strong made hand facing a bet"},
    },
    {
        "state": {
            "position": "CO",
            "street": "turn",
            "hole_cards": ["AD", "5D"],
            "board_cards": ["KD", "8D", "2C", "9S"],
            "pot": 16.0,
            "to_call": 5.0,
            "stack": 52.0,
            "min_raise": 15.0,
            "player_count": 5,
            "betting_history": [{"position": "MP", "action": "bet", "amount": 5.0, "street": "turn"}],
            "derived": {"pot_odds": 0.238095, "spr": 2.47619, "strength_proxy": 0.689286, "hole_cards_missing": False},
        },
        "output": {"action": "call", "confidence": 0.59, "rationale": "draw equity with acceptable price but not enough made strength to raise"},
    },
    {
        "state": {
            "position": "UTG",
            "street": "preflop",
            "hole_cards": ["QS", "7C"],
            "board_cards": [],
            "pot": 1.5,
            "to_call": 0.0,
            "stack": 120.0,
            "min_raise": 2.5,
            "player_count": 6,
            "betting_history": [],
            "derived": {"pot_odds": 0.0, "spr": 80.0, "strength_proxy": 0.596429, "hole_cards_missing": False},
        },
        "output": {"action": "fold", "confidence": 0.50, "rationale": "marginal offsuit hand from early position"},
    },
    {
        "state": {
            "position": "BTN",
            "street": "river",
            "hole_cards": ["AS", "AC"],
            "board_cards": ["AH", "KD", "7C", "7S", "2D"],
            "pot": 45.0,
            "to_call": 0.0,
            "stack": 110.0,
            "min_raise": 22.5,
            "player_count": 3,
            "betting_history": [{"position": "BB", "action": "check", "amount": 0.0, "street": "river"}],
            "derived": {"pot_odds": 0.0, "spr": 2.444444, "strength_proxy": 1.0, "hole_cards_missing": False},
        },
        "output": {"action": "bet", "confidence": 0.81, "rationale": "very strong made hand can value bet when checked to"},
    },
)


@dataclass(frozen=True)
class DecisionPromptConfig:
    profile: str = "rules_zero_shot"
    few_shot_count: int = 0
    include_rules: bool = True
    include_guidelines: bool = True


def state_payload(request: PredictionRequest) -> dict[str, Any]:
    features = request_to_features(request)
    return {
        "position": request.position,
        "street": request.street,
        "hole_cards": request.hole_cards,
        "board_cards": request.board_cards,
        "pot": request.pot,
        "to_call": request.to_call,
        "stack": request.stack,
        "min_raise": request.min_raise,
        "player_count": request.player_count,
        "betting_history": request.betting_history[-12:],
        "derived": {
            "pot_odds": round(features.get("pot_odds", 0.0), 6),
            "spr": round(features.get("spr", 0.0), 6),
            "strength_proxy": round(features.get("strength_proxy", 0.0), 6),
            "street_aggression_ratio": round(
                features.get("street_aggression_ratio", features.get("hist_aggression_ratio", 0.0)),
                6,
            ),
            "hole_cards_missing": bool(features.get("hole_cards_missing", 0.0)),
        },
    }


def prompt_config_for_profile(profile: str, few_shot_count: int = 0) -> DecisionPromptConfig:
    if profile == "minimal":
        return DecisionPromptConfig(profile=profile, few_shot_count=0, include_rules=False, include_guidelines=True)
    if profile == "rules_zero_shot":
        return DecisionPromptConfig(profile=profile, few_shot_count=0, include_rules=True, include_guidelines=True)
    if profile == "rules_few_shot":
        return DecisionPromptConfig(
            profile=profile,
            few_shot_count=few_shot_count or 5,
            include_rules=True,
            include_guidelines=True,
        )
    if profile == "candidate_ranker":
        return DecisionPromptConfig(
            profile=profile,
            few_shot_count=few_shot_count,
            include_rules=True,
            include_guidelines=True,
        )
    raise ValueError(f"Unsupported decision prompt profile: {profile}")


def render_numbered_lines(title: str, lines: tuple[str, ...]) -> str:
    rendered = [f"{title}:"]
    rendered.extend(f"{index}. {line}" for index, line in enumerate(lines, start=1))
    return "\n".join(rendered)


def render_examples(count: int) -> str:
    selected = FEW_SHOT_EXAMPLES[: max(0, min(count, len(FEW_SHOT_EXAMPLES)))]
    if not selected:
        return ""
    sections: list[str] = ["Examples:"]
    for index, example in enumerate(selected, start=1):
        sections.append(f"Example {index} state:")
        sections.append(json.dumps(example["state"], sort_keys=True))
        sections.append("Example output:")
        sections.append(json.dumps(example["output"], sort_keys=True))
    return "\n".join(sections)


def build_contextual_decision_prompt(
    request: PredictionRequest,
    *,
    allowed_actions: tuple[str, ...],
    profile: str = "rules_zero_shot",
    few_shot_count: int = 0,
    candidates: tuple[str, ...] | None = None,
) -> str:
    config = prompt_config_for_profile(profile, few_shot_count=few_shot_count)
    state = state_payload(request)
    sections = [
        "Task: classify one poker decision from a structured game state.",
        "The output must be compact JSON only.",
        f"Allowed actions: {', '.join(allowed_actions)}.",
        "Output schema:",
        json.dumps(DECISION_OUTPUT_SCHEMA, sort_keys=True),
    ]
    if config.include_rules:
        sections.append(render_numbered_lines("Poker rules and decision context", POKER_RULES))
    if config.include_guidelines:
        sections.append(render_numbered_lines("Formal constraints", FORMAL_GUIDELINES))
    if candidates is not None:
        sections.append("Candidate actions:")
        sections.append(json.dumps(list(candidates), sort_keys=True))
        sections.append("Select the best candidate action. Do not introduce an action outside the candidate list.")
    examples = render_examples(config.few_shot_count)
    if examples:
        sections.append(examples)
    sections.append("Current state:")
    sections.append(json.dumps(state, sort_keys=True))
    sections.append("JSON:")
    return "\n\n".join(sections)


def describe_prompt_profile(profile: str, few_shot_count: int = 0) -> dict[str, Any]:
    config = prompt_config_for_profile(profile, few_shot_count=few_shot_count)
    return {
        "profile": config.profile,
        "include_rules": config.include_rules,
        "include_guidelines": config.include_guidelines,
        "few_shot_count": config.few_shot_count,
        "rules_count": len(POKER_RULES) if config.include_rules else 0,
        "guidelines_count": len(FORMAL_GUIDELINES) if config.include_guidelines else 0,
        "output_schema": DECISION_OUTPUT_SCHEMA,
    }
