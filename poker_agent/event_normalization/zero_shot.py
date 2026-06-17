from __future__ import annotations

from poker_agent.event_normalization.backends import TextModelBackend
from poker_agent.event_normalization.prompts import build_zero_shot_prompt
from poker_agent.event_normalization.schema import ExtractionResult, parse_event_json


class ZeroShotExtractor:
    def __init__(self, backend: TextModelBackend) -> None:
        self.backend = backend

    def extract(self, text: str) -> ExtractionResult:
        raw_output = self.backend.generate_json(build_zero_shot_prompt(text))
        return parse_event_json(raw_output)
