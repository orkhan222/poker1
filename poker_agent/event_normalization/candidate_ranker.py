from __future__ import annotations

import json

from poker_agent.event_normalization.backends import TextModelBackend
from poker_agent.event_normalization.parser import DeterministicParser
from poker_agent.event_normalization.prompts import build_candidate_ranking_prompt
from poker_agent.event_normalization.schema import ExtractionResult, Event, model_dump_event


class CandidateRanker:
    def __init__(self, backend: TextModelBackend, parser: DeterministicParser | None = None) -> None:
        self.backend = backend
        self.parser = parser or DeterministicParser()

    def extract(self, text: str) -> ExtractionResult:
        candidates = self.parser.generate_candidates(text)
        candidate_payloads = [candidate.to_payload(index) for index, candidate in enumerate(candidates)]
        raw_output = self.backend.rank_candidate(build_candidate_ranking_prompt(text, candidates), candidate_payloads)
        try:
            payload = json.loads(raw_output)
            candidate_id = int(payload.get("candidate_id"))
            confidence = float(payload.get("confidence", 0.0))
            selected = candidates[candidate_id].event
        except (ValueError, TypeError, IndexError, json.JSONDecodeError, AttributeError) as exc:
            return ExtractionResult(
                event=Event(event_type="unmatched", confidence=0.0),
                raw_output=raw_output,
                schema_valid=False,
                errors=[str(exc)],
                candidate_count=len(candidates),
            )
        event_payload = model_dump_event(selected)
        event_payload["confidence"] = max(0.0, min(1.0, confidence))
        return ExtractionResult(
            event=Event(**event_payload),
            raw_output=raw_output,
            schema_valid=True,
            candidate_count=len(candidates),
        )
