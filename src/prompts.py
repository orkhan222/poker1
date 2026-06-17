from __future__ import annotations


SCHEMA_DESCRIPTION = """{
  "event_type": "player_action | card_event | pot_event | dealer_event | unmatched",
  "player": "string | null",
  "action": "fold | check | call | bet | raise | all_in | small_blind | big_blind | null",
  "amount": "float | null",
  "cards": ["string"] | null,
  "confidence": "float between 0 and 1"
}"""

INSTRUCTION_TEMPLATE = """You are an event-normalization model.

Convert the noisy poker OCR/dealer log into valid JSON only.

Rules:
- Return JSON only.
- Do not explain.
- Do not add extra keys.
- If uncertain, return event_type = "unmatched".
- Confidence must be between 0 and 1.

Schema:
{schema}

Raw text:
{raw_text}

JSON:
"""


def build_prompt(raw_text: str) -> str:
    return INSTRUCTION_TEMPLATE.format(schema=SCHEMA_DESCRIPTION, raw_text=str(raw_text).strip())

