# LLM Event-Normalization Architecture

## Decision

The LLM component in this project is not an autonomous poker-playing agent. It is a constrained event-normalization layer inside the data pipeline. Its responsibility is to convert noisy OCR and dealer-log text into validated JSON events that can be consumed by downstream training and evaluation jobs.

The LLM must not decide poker strategy directly in production. Poker action prediction remains the responsibility of the supervised policy model and later simulation-trained policy models. The LLM is used to improve data quality where deterministic parsing is insufficient.

## Target Contract

Every extracted event must conform to the versioned event schema in `poker_agent/event_schema.py`.

Canonical event types:

- `player_action`
- `card_update`
- `stack_update`
- `unmatched`

Example normalized event:

```json
{
  "event_type": "player_action",
  "player": "Player3",
  "action": "raise",
  "amount": 4.5,
  "cards": [],
  "raw_text": "Player3 raises to 4.5",
  "confidence": 0.91,
  "schema_version": "event_schema_v1"
}
```

Low-confidence or invalid outputs are returned as `unmatched` rather than being forced into a misleading class.

## Recommended Architecture

The production candidate is a schema-routed hybrid architecture.

1. Deterministic parser handles stable, well-known event formats.
2. Ambiguity detector routes corrupted, incomplete, or OCR-noisy text to the LLM path.
3. Candidate generator builds a small set of valid event candidates.
4. LLM ranks candidates instead of generating unconstrained free-form output.
5. JSON schema validator verifies all outputs.
6. Confidence gate returns `unmatched` when validation or confidence fails.

This architecture is safer than a free-form generative agent because the output space is bounded, validation is explicit, and failure modes are observable.

## Alternatives Considered

| Approach | Strength | Weakness | Recommendation |
| --- | --- | --- | --- |
| Free-form JSON generation | Simple to prototype | Higher invalid-output risk, harder to debug | Not recommended for production |
| Direct poker decision LLM | Easy to demo | Does not solve missing-card and imbalance issues | Not recommended now |
| Deterministic parser only | Fast and cheap | Brittle on OCR corruption | Use as first stage |
| Candidate ranker | Controlled output, measurable | Requires candidate generation | Recommended |
| Schema-routed hybrid | Best reliability/cost balance | More engineering work | Primary architecture |

## Model Strategy

The first evaluation track should use 1B-2B instruction models because the task is constrained extraction and candidate ranking, not open-ended reasoning.

| Model | Role | Notes |
| --- | --- | --- |
| Qwen2.5-1.5B-Instruct | Primary LoRA candidate | Strong structured-output baseline and practical deployment size |
| Qwen3-1.7B | Primary alternative | Good candidate for extraction and tool-style workflows |
| SmolLM2-1.7B-Instruct | Lightweight baseline | Useful for local benchmark continuity |
| Gemma 3 1B IT | Resource-constrained fallback | Lower memory target, expected lower ceiling |
| Llama 3.x 1B | Secondary option | License and lifecycle must be checked before default use |

The expected production path is Qwen-family extraction or ranking behind the deterministic parser. SmolLM remains useful as a lightweight baseline, not as the default final model.

## LoRA Recommendation

LoRA or QLoRA should be applied to the extraction layer first, not to the poker policy model.

Expected benefits:

- higher schema-validity rate;
- fewer collapsed predictions;
- stronger robustness to OCR corruption;
- better action, card, player, and amount extraction;
- lower deployment cost than larger general-purpose models.

Risks:

- overfitting if the reviewed dataset remains too small;
- misleading metrics if evaluation examples are too easy;
- no direct fix for missing hole cards or class imbalance in the policy model;
- latency and memory must be measured on the target deployment machine.

Initial QLoRA configuration:

```yaml
base_model: Qwen2.5-1.5B-Instruct
method: qlora
quantization: 4bit_nf4
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 0.0002
epochs: 3
max_seq_len: 2048
metrics:
  - macro_f1
  - schema_validity_rate
  - action_exact_match
  - card_exact_match
  - amount_mae
```

Minimum useful dataset sizes:

| Dataset Size | Purpose |
| ---: | --- |
| 500-1,000 labeled examples | LoRA smoke test |
| 5,000+ labeled examples | reliable model comparison |
| 20,000+ labeled examples | production candidate evaluation |

## Evaluation

The LLM event-normalization track is evaluated separately from poker decision quality.

Primary metrics:

- macro F1 over event type;
- action exact match;
- card exact match;
- amount mean absolute error;
- schema-validity rate;
- unmatched rate;
- fallback rate from deterministic parser to LLM;
- latency p50 and p95;
- memory footprint.

The existing Phase 1 dataset builder creates a versioned schema, corrupted OCR variants, and grouped train/validation/test splits. Current Phase 1 artifact summary:

| Metric | Value |
| --- | ---: |
| reviewed parent examples | 24 |
| expanded examples | 120 |
| grouped examples | 24 |
| train rows | 75 |
| validation rows | 20 |
| test rows | 25 |

## Implementation Roadmap

### Phase 1: Data and Schema

Status: implemented.

Deliverables:

- final event schema;
- reviewed gold dataset expansion;
- corrupted OCR examples;
- grouped train/validation/test splits;
- schema and dataset validation reports.

### Phase 2: Baselines

Benchmark deterministic parser, zero-shot prompting, few-shot prompting, transformer extraction, and candidate ranking on the same grouped splits.

Expected duration: 3-4 engineering days.

### Phase 3: LoRA Fine-Tuning

Fine-tune Qwen2.5-1.5B-Instruct with QLoRA for event extraction or candidate ranking. Compare against deterministic and non-fine-tuned baselines.

Expected duration: 4-6 engineering days after enough reviewed labels are available.

### Phase 4: Validation

Evaluate macro F1, schema validity, exact-match metrics, latency, memory usage, and failure cases. Produce a model card and acceptance report.

Expected duration: 3-4 engineering days.

### Phase 5: Deployment

Integrate the selected model as an optional fallback extraction service behind the deterministic parser. Keep the API contract schema-first and return `unmatched` on low-confidence outputs.

Expected duration: 3-5 engineering days.

## Recommendation

Proceed with the schema-routed hybrid architecture. It directly addresses the reviewer concern about noisy OCR and weak data quality while keeping the LLM component bounded, measurable, and production-reviewable.

The next technical milestone should be a Qwen2.5-1.5B candidate-ranker experiment using the Phase 1 grouped splits. The acceptance target for the next milestone is improved macro F1 and schema-validity rate versus the current deterministic and lightweight transformer baselines, with latency and memory measurements included.
