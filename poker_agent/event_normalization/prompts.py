from __future__ import annotations

import json
from typing import Any

from poker_agent.event_normalization.parser import Candidate
from poker_agent.event_normalization.schema import model_dump_event


SCHEMA_DESCRIPTION = """
Return one JSON object with exactly these fields:
{
  "event_type": "player_action | card_event | pot_event | dealer_event | unmatched",
  "player": "string or null",
  "action": "fold | check | call | bet | raise | all_in | small_blind | big_blind | null",
  "amount": "number or null",
  "cards": ["AS", "KD"] or [],
  "confidence": "number between 0 and 1"
}
""".strip()

ZERO_SHOT_SYSTEM = """
You normalize noisy poker OCR and dealer-log text into a strict JSON event.
Do not explain the answer. Do not return markdown. Return JSON only.
""".strip()

FEW_SHOT_SYSTEM = """
You normalize noisy poker OCR and dealer-log text into a strict JSON event.
Use the examples as formatting and OCR-correction references. Return JSON only.
""".strip()

CANDIDATE_RANKING_SYSTEM = """
You are given noisy poker OCR text and a list of candidate normalized events.
Choose the most likely candidate. Return JSON only:
{"candidate_id": 0, "confidence": 0.93}
""".strip()


def build_zero_shot_prompt(raw_text: str) -> str:
    return "\n\n".join([ZERO_SHOT_SYSTEM, SCHEMA_DESCRIPTION, f"RAW_TEXT: {raw_text}", "JSON:"])


def build_few_shot_prompt(raw_text: str, examples: list[dict[str, Any]]) -> str:
    formatted_examples = []
    for example in examples:
        expected = example["expected"]
        formatted_examples.append(
            "RAW_TEXT: "
            + str(example["raw_text"])
            + "\nJSON: "
            + json.dumps(model_dump_event(expected), sort_keys=True)
        )
    return "\n\n".join(
        [
            FEW_SHOT_SYSTEM,
            SCHEMA_DESCRIPTION,
            "Examples:",
            "\n\n".join(formatted_examples),
            f"RAW_TEXT: {raw_text}",
            "JSON:",
        ]
    )


def build_candidate_ranking_prompt(raw_text: str, candidates: list[Candidate]) -> str:
    payload = [candidate.to_payload(index) for index, candidate in enumerate(candidates)]
    return "\n\n".join(
        [
            CANDIDATE_RANKING_SYSTEM,
            f"RAW_TEXT: {raw_text}",
            "CANDIDATES:",
            json.dumps(payload, indent=2, sort_keys=True),
            "JSON:",
        ]
    )
