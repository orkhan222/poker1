from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Protocol

from poker_agent.event_normalization.parser import DeterministicParser
from poker_agent.event_normalization.schema import model_dump_event


class TextModelBackend(Protocol):
    model_id: str

    def generate_json(self, prompt: str) -> str:
        ...

    def rank_candidate(self, prompt: str, candidates: list[dict[str, Any]]) -> str:
        ...


def raw_text_from_prompt(prompt: str) -> str:
    matches = re.findall(r"RAW_TEXT:\s*(.+)", prompt)
    return matches[-1].strip() if matches else prompt


class HeuristicTextBackend:
    def __init__(self, *, model_id: str, parser_profile: str = "full", seed: int = 42) -> None:
        self.model_id = model_id
        self.parser = DeterministicParser(ocr_profile=parser_profile)
        self.random = random.Random(seed)

    def generate_json(self, prompt: str) -> str:
        event = self.parser.parse(raw_text_from_prompt(prompt))
        return json.dumps(model_dump_event(event), sort_keys=True)

    def rank_candidate(self, prompt: str, candidates: list[dict[str, Any]]) -> str:
        if not candidates:
            return json.dumps({"candidate_id": 0, "confidence": 0.0})
        best_index = 0
        best_score = -1.0
        for index, candidate in enumerate(candidates):
            confidence = float(candidate.get("confidence") or 0.0)
            event_type_bonus = 0.05 if candidate.get("event_type") != "unmatched" else 0.0
            score = confidence + event_type_bonus
            if score > best_score:
                best_index = index
                best_score = score
        return json.dumps({"candidate_id": best_index, "confidence": min(0.99, max(0.0, best_score))})


class TransformersTextBackend:
    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers before using the transformers backend") from exc

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_json(self, prompt: str) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def rank_candidate(self, prompt: str, candidates: list[dict[str, Any]]) -> str:
        return self.generate_json(prompt)


def create_backend(
    *,
    backend: str,
    model_id: str,
    parser_profile: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> TextModelBackend:
    if backend == "heuristic":
        return HeuristicTextBackend(model_id=model_id, parser_profile=parser_profile, seed=seed)
    if backend == "transformers":
        return TransformersTextBackend(
            model_id=model_id,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
    raise ValueError(f"Unsupported backend: {backend}")
