from __future__ import annotations

from typing import Any

from poker_agent.event_normalization.backends import TextModelBackend
from poker_agent.event_normalization.prompts import build_few_shot_prompt
from poker_agent.event_normalization.schema import ExtractionResult, parse_event_json


class FewShotExtractor:
    def __init__(self, backend: TextModelBackend, examples: list[dict[str, Any]]) -> None:
        self.backend = backend
        self.examples = examples

    def extract(self, text: str) -> ExtractionResult:
        raw_output = self.backend.generate_json(build_few_shot_prompt(text, self.examples))
        return parse_event_json(raw_output)
