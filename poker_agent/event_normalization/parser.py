from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - used only when optional dependency is absent
    fuzz = None
    process = None

from difflib import SequenceMatcher

from poker_agent.event_normalization.schema import (
    ACTIONS,
    Event,
    build_event,
    model_dump_event,
    normalize_amount,
    normalize_card,
)


ACTION_TERMS = {
    "fold": ("fold", "folds", "fo1d", "f0ld"),
    "check": ("check", "checks", "chek", "chcck"),
    "call": ("call", "calls", "cail", "ca11", "cal1"),
    "bet": ("bet", "bets", "bct"),
    "raise": ("raise", "raises", "raised", "ra1se", "ralse", "rais"),
    "all_in": ("all in", "all-in", "allin", "a11 in"),
    "small_blind": ("small blind", "small_blind", "posts sb", "sb"),
    "big_blind": ("big blind", "big_blind", "posts bb", "bb"),
}

OCR_REPLACEMENTS = (
    (r"\bp1ayer\b", "player"),
    (r"\bplyr\b", "player"),
    (r"\bra1se\b", "raise"),
    (r"\bralse\b", "raise"),
    (r"\brais\b", "raise"),
    (r"\bcail\b", "call"),
    (r"\bca11\b", "call"),
    (r"\bcal1\b", "call"),
    (r"\bca1ls\b", "calls"),
    (r"\bca1l\b", "call"),
    (r"\bf0ld\b", "fold"),
    (r"\bfo1d\b", "fold"),
    (r"\bchek\b", "check"),
    (r"\bch3ck\b", "check"),
    (r"\bp0st\b", "post"),
    (r"\bb1ind\b", "blind"),
)

CARD_RE = re.compile(r"\b(?:10|[2-9TJQKA])\s*[CDHS♠♤♥♡♦♢♣♧]\b", flags=re.IGNORECASE)
UNICODE_CARD_RE = re.compile(r"(?:10|[2-9TJQKA])\s*[♠♤♥♡♦♢♣♧]", flags=re.IGNORECASE)
AMOUNT_RE = re.compile(r"(?:\$|bb\s*)?(-?\d+(?:[\.,]\d+)?)")
PLAYER_RE = re.compile(r"\b(?:player|plyr|p1ayer)\s*[_-]?\s*(\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    event: Event
    reason: str

    def to_payload(self, candidate_id: int) -> dict[str, Any]:
        payload = model_dump_event(self.event)
        payload["candidate_id"] = candidate_id
        payload["reason"] = self.reason
        return payload


def fuzzy_score(value: str, choices: tuple[str, ...]) -> tuple[str | None, float]:
    if not value:
        return None, 0.0
    if process is not None and fuzz is not None:
        match = process.extractOne(value, choices, scorer=fuzz.WRatio)
        if not match:
            return None, 0.0
        return str(match[0]), float(match[1]) / 100.0
    scored = [(choice, SequenceMatcher(None, value, choice).ratio()) for choice in choices]
    return max(scored, key=lambda item: item[1])


class DeterministicParser:
    def __init__(self, *, fuzzy_threshold: float = 0.82, ocr_profile: str = "full") -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.ocr_profile = ocr_profile

    def parse(self, text: str) -> Event:
        normalized = self.normalize_text(text)
        player = self.extract_player(normalized)
        cards = self.extract_cards(text)
        action, action_confidence = self.extract_action(normalized)
        amount = self.extract_amount(normalized)

        if action:
            return build_event(
                {
                    "event_type": "player_action",
                    "player": player,
                    "action": action,
                    "amount": amount,
                    "cards": [],
                    "confidence": min(0.99, max(0.55, action_confidence)),
                }
            )
        if cards:
            return build_event(
                {
                    "event_type": "card_event",
                    "player": player,
                    "cards": cards,
                    "confidence": 0.92,
                }
            )
        if self.looks_like_dealer_event(normalized):
            return build_event(
                {
                    "event_type": "dealer_event",
                    "player": player,
                    "amount": amount,
                    "confidence": 0.75 if amount is not None else 0.62,
                }
            )
        if self.looks_like_pot_event(normalized):
            return build_event(
                {
                    "event_type": "pot_event",
                    "player": player,
                    "amount": amount,
                    "confidence": 0.82 if amount is not None else 0.60,
                }
            )
        return Event(event_type="unmatched", confidence=0.25)

    def generate_candidates(self, text: str) -> list[Candidate]:
        primary = self.parse(text)
        candidates = [Candidate(primary, "deterministic_primary")]
        amount = primary.amount or self.extract_amount(self.normalize_text(text))
        player = primary.player or self.extract_player(self.normalize_text(text))

        if primary.event_type == "player_action":
            alternatives = self.neighbor_actions(primary.action)
            for action in alternatives:
                candidates.append(
                    Candidate(
                        build_event(
                            {
                                "event_type": "player_action",
                                "player": player,
                                "action": action,
                                "amount": amount,
                                "confidence": 0.45,
                            }
                        ),
                        f"neighbor_action_{action}",
                    )
                )
        if amount is not None:
            candidates.append(
                Candidate(
                    build_event(
                        {
                            "event_type": "pot_event",
                            "player": player,
                            "amount": amount,
                            "confidence": 0.40,
                        }
                    ),
                    "amount_as_pot_event",
                )
            )
        candidates.append(Candidate(Event(event_type="unmatched", confidence=0.20), "fallback_unmatched"))
        return self.deduplicate_candidates(candidates)

    def normalize_text(self, text: str) -> str:
        normalized = str(text or "").replace("\u00a0", " ").strip().lower()
        normalized = normalized.replace("♠", " s").replace("♤", " s")
        normalized = normalized.replace("♥", " h").replace("♡", " h")
        normalized = normalized.replace("♦", " d").replace("♢", " d")
        normalized = normalized.replace("♣", " c").replace("♧", " c")
        if self.ocr_profile == "full":
            for pattern, replacement in OCR_REPLACEMENTS:
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"[_:/|]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def extract_player(self, normalized: str) -> str | None:
        match = PLAYER_RE.search(normalized)
        if match:
            return f"Player{match.group(1)}"
        return None

    def extract_action(self, normalized: str) -> tuple[str | None, float]:
        for action, terms in ACTION_TERMS.items():
            for term in terms:
                if re.search(rf"\b{re.escape(term)}\b", normalized):
                    return action, 0.95
        tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
        for token in tokens:
            canonical, score = fuzzy_score(token, tuple(term for terms in ACTION_TERMS.values() for term in terms))
            if canonical and score >= self.fuzzy_threshold:
                for action, terms in ACTION_TERMS.items():
                    if canonical in terms:
                        return action, min(0.88, score)
        return None, 0.0

    def extract_cards(self, text: str) -> list[str]:
        raw_matches = CARD_RE.findall(text) + UNICODE_CARD_RE.findall(text)
        cards: list[str] = []
        for raw in raw_matches:
            card = normalize_card(raw)
            if card and card not in cards:
                cards.append(card)
        return cards

    def extract_amount(self, normalized: str) -> float | None:
        money = re.search(r"\$\s*(-?\d+(?:[\.,]\d+)?)", normalized)
        if money:
            return normalize_amount(money.group(1))
        stack_pair = re.search(
            r"\bstack\b[^\d-]{0,12}(-?\d+(?:[\.,]\d+)?)[^\d-]+(-?\d+(?:[\.,]\d+)?)",
            normalized,
        )
        if stack_pair:
            return normalize_amount(stack_pair.group(2))
        prioritized = re.search(
            r"\b(?:amount|to|pot|stack|diff|blind|post|sb|bb|bet|raise|call|wins?|payout)\b[^\d-]{0,12}(-?\d+(?:[\.,]\d+)?)",
            normalized,
        )
        if prioritized:
            return normalize_amount(prioritized.group(1))
        return None

    def looks_like_pot_event(self, normalized: str) -> bool:
        return any(term in normalized for term in ("pot", "stack", "chip", "diff", "balance"))

    def looks_like_dealer_event(self, normalized: str) -> bool:
        return any(term in normalized for term in ("dealer", "winner", "wins", "collected", "payout", "shows"))

    def neighbor_actions(self, action: str | None) -> list[str]:
        if action == "raise":
            return ["bet", "call", "all_in"]
        if action == "bet":
            return ["raise", "call"]
        if action == "call":
            return ["check", "raise"]
        if action == "check":
            return ["call", "fold"]
        if action == "fold":
            return ["check", "call"]
        return [action for action in ACTIONS if action not in {"small_blind", "big_blind"}][:3]

    def deduplicate_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        seen: set[tuple[Any, ...]] = set()
        unique: list[Candidate] = []
        for candidate in candidates:
            payload = model_dump_event(candidate.event)
            key = (
                payload["event_type"],
                payload.get("player"),
                payload.get("action"),
                payload.get("amount"),
                tuple(payload.get("cards") or []),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique
