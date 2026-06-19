from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from poker_agent.action_planning import build_action_plan
from poker_agent.decision_context import build_contextual_decision_prompt, state_payload
from poker_agent.features import normalize_action, request_to_features
from poker_agent.schemas import PredictionRequest, PredictionResponse


DECISION_ACTIONS = ("fold", "call", "check", "bet", "raise")


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {key: math.exp(value - max_score) for key, value in scores.items()}
    total = sum(exp_scores.values()) or 1.0
    return {key: value / total for key, value in exp_scores.items()}


def _state_payload(request: PredictionRequest) -> dict[str, Any]:
    return state_payload(request)


def build_decision_prompt(
    request: PredictionRequest,
    allowed_actions: tuple[str, ...] = DECISION_ACTIONS,
    *,
    prompt_profile: str = "rules_zero_shot",
    few_shot_count: int = 0,
    candidates: tuple[str, ...] | None = None,
) -> str:
    return build_contextual_decision_prompt(
        request,
        allowed_actions=allowed_actions,
        profile=prompt_profile,
        few_shot_count=few_shot_count,
        candidates=candidates,
    )


def normalize_decision_action(raw: Any) -> str | None:
    action = normalize_action(str(raw or ""))
    if action == "all_in":
        return "raise"
    if action in DECISION_ACTIONS:
        return action
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def probabilities_from_action(action: str, confidence: float, allowed_actions: tuple[str, ...]) -> dict[str, float]:
    confidence = min(max(float(confidence), 0.0), 1.0)
    if action not in allowed_actions:
        action = allowed_actions[0]
        confidence = 0.0
    remaining = max(1, len(allowed_actions) - 1)
    spill = (1.0 - confidence) / remaining
    probabilities = {candidate: spill for candidate in allowed_actions}
    probabilities[action] = confidence
    total = sum(probabilities.values()) or 1.0
    return {key: value / total for key, value in probabilities.items()}


@dataclass
class ParsedDecision:
    action: str
    probabilities: dict[str, float]
    confidence: float
    raw_text: str
    valid: bool


def parse_decision_text(text: str, allowed_actions: tuple[str, ...] = DECISION_ACTIONS) -> ParsedDecision:
    payload = _extract_json_object(text)
    if payload:
        action = normalize_decision_action(payload.get("action") or payload.get("decision"))
        raw_probabilities = payload.get("probabilities")
        if isinstance(raw_probabilities, dict):
            probabilities: dict[str, float] = {}
            for key, value in raw_probabilities.items():
                normalized = normalize_decision_action(key)
                if normalized in allowed_actions:
                    try:
                        probabilities[normalized] = float(value)
                    except (TypeError, ValueError):
                        pass
            total = sum(value for value in probabilities.values() if value > 0.0)
            if total > 0.0:
                probabilities = {key: max(value, 0.0) / total for key, value in probabilities.items()}
                action = action or max(probabilities, key=probabilities.get)
                return ParsedDecision(
                    action=action,
                    probabilities={candidate: probabilities.get(candidate, 0.0) for candidate in allowed_actions},
                    confidence=max(probabilities.values(), default=0.0),
                    raw_text=text,
                    valid=action in allowed_actions,
                )
        if action in allowed_actions:
            confidence = payload.get("confidence", 0.65)
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.65
            return ParsedDecision(
                action=action,
                probabilities=probabilities_from_action(action, confidence_value, allowed_actions),
                confidence=confidence_value,
                raw_text=text,
                valid=True,
            )

    match = re.search(r"\b(fold|call|check|bet|raise|all[-_\s]?in)\b", text, flags=re.IGNORECASE)
    action = normalize_decision_action(match.group(1)) if match else None
    if action in allowed_actions:
        return ParsedDecision(
            action=action,
            probabilities=probabilities_from_action(action, 0.55, allowed_actions),
            confidence=0.55,
            raw_text=text,
            valid=True,
        )
    return ParsedDecision(
        action="fold",
        probabilities=probabilities_from_action("fold", 0.0, allowed_actions),
        confidence=0.0,
        raw_text=text,
        valid=False,
    )


class DecisionTextProvider(Protocol):
    name: str

    def generate(self, prompt: str, request: PredictionRequest) -> str:
        ...

    def score_candidates(
        self,
        prompt: str,
        request: PredictionRequest,
        candidates: tuple[str, ...],
    ) -> dict[str, float]:
        ...


@dataclass
class HeuristicTextProvider:
    name: str = "heuristic_text"

    def _scores(self, request: PredictionRequest) -> dict[str, float]:
        features = request_to_features(request)
        strength = features.get("strength_proxy", 0.0)
        pot_odds = features.get("pot_odds", 0.0)
        spr = min(features.get("spr", 0.0), 20.0) / 20.0
        aggression = features.get("hist_aggression_ratio", 0.0)
        facing_bet = request.to_call > 0
        can_check = request.to_call <= 0

        scores = {
            "fold": -0.4 - 1.8 * strength + 1.8 * pot_odds + 0.7 * aggression,
            "call": 0.2 + 1.0 * strength - 1.2 * pot_odds + 0.2 * aggression,
            "check": 1.2 if can_check else -2.4,
            "bet": -0.7 + 1.6 * strength + 0.4 * spr - (0.6 if facing_bet else 0.0),
            "raise": -0.9 + 2.3 * strength - 0.5 * pot_odds + 0.3 * spr,
        }
        if len(request.hole_cards) < 2:
            scores["fold"] += 0.5 if facing_bet else 0.0
            scores["raise"] -= 0.45
            scores["bet"] -= 0.25
        if request.street == "preflop" and strength >= 0.65:
            scores["raise"] += 0.55
        return scores

    def score_candidates(
        self,
        prompt: str,
        request: PredictionRequest,
        candidates: tuple[str, ...],
    ) -> dict[str, float]:
        scores = self._scores(request)
        return {candidate: scores.get(candidate, -10.0) for candidate in candidates}

    def generate(self, prompt: str, request: PredictionRequest) -> str:
        scores = self._scores(request)
        probabilities = _softmax(scores)
        action = max(probabilities, key=probabilities.get)
        return json.dumps(
            {
                "action": action,
                "confidence": round(probabilities[action], 6),
                "probabilities": {key: round(value, 6) for key, value in probabilities.items()},
                "rationale": "selected from structured state features",
            },
            sort_keys=True,
        )


@dataclass
class TransformersTextProvider:
    model_id: str
    device: str = "cpu"
    max_new_tokens: int = 64
    temperature: float = 0.0
    name: str = field(init=False, default="transformers")
    _tokenizer: Any = field(init=False, default=None)
    _model: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to run the local text-model provider") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        self._model.eval()

    def _encode_prompt(self, prompt: str) -> Any:
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        return tokenizer(prompt, return_tensors="pt").input_ids

    def generate(self, prompt: str, request: PredictionRequest) -> str:
        del request
        torch = self._torch
        inputs = self._encode_prompt(prompt)
        if self.device != "cpu":
            inputs = inputs.to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs.shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def score_candidates(
        self,
        prompt: str,
        request: PredictionRequest,
        candidates: tuple[str, ...],
    ) -> dict[str, float]:
        del prompt
        return HeuristicTextProvider().score_candidates("", request, candidates)


@dataclass
class LLMDecisionAgent:
    provider: DecisionTextProvider
    mode: str = "candidate_ranker"
    allowed_actions: tuple[str, ...] = DECISION_ACTIONS
    prompt_profile: str = "rules_zero_shot"
    few_shot_count: int = 0

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        prompt = build_decision_prompt(
            request,
            self.allowed_actions,
            prompt_profile=self.prompt_profile,
            few_shot_count=self.few_shot_count,
            candidates=self.allowed_actions if self.mode == "candidate_ranker" else None,
        )
        started = time.perf_counter()
        warnings: list[str] = []
        raw_text = ""

        if self.mode == "candidate_ranker":
            scores = self.provider.score_candidates(prompt, request, self.allowed_actions)
            probabilities = _softmax(scores)
            action = max(probabilities, key=probabilities.get)
            confidence = probabilities[action]
        elif self.mode == "freeform":
            raw_text = self.provider.generate(prompt, request)
            parsed = parse_decision_text(raw_text, self.allowed_actions)
            action = parsed.action
            probabilities = parsed.probabilities
            confidence = parsed.confidence
            if not parsed.valid:
                warnings.append("Text output did not contain a valid action; fold fallback was used.")
        else:
            raise ValueError(f"Unsupported decision mode: {self.mode}")

        latency_ms = (time.perf_counter() - started) * 1000.0
        plan = build_action_plan(
            request,
            action,
            confidence=confidence,
            processing_time_ms=latency_ms,
        )
        response = PredictionResponse(
            action=action,
            probabilities=probabilities,
            confidence=confidence,
            bet_size=plan.bet_size,
            wait_time_ms=plan.wait_time_ms,
            sizing_method=plan.sizing_method,
            timing_method=plan.timing_method,
            model_status=f"{self.provider.name}:{self.mode}",
            warnings=warnings,
        )
        response.raw_text = raw_text
        response.latency_ms = latency_ms
        response.prompt = prompt
        return response


def request_to_jsonable(request: PredictionRequest) -> dict[str, Any]:
    return asdict(request)
