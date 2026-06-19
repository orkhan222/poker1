# LLM Agent Architecture

## Scope

The LLM component is a bounded event-normalization and simulation-preparation layer. It is not an autonomous poker policy and it is not responsible for independent strategic play.

## Selected Architecture

The selected production candidate is `hybrid_parser_qlora`.

Pipeline:

```text
OCR / dealer log
-> deterministic parser
-> candidate generation
-> QLoRA-backed fallback only when needed
-> JSON schema validation
-> normalized event stream
-> simulation event export
-> downstream poker policy training
```

## Why This Architecture

The current bottleneck is data quality: noisy OCR, corrupted dealer messages, incomplete event reconstruction, and missing cards. A fully generative poker-playing model would not solve these issues and would create unnecessary operational risk.

The hybrid architecture keeps deterministic behavior for high-confidence events and reserves the LLM adapter for ambiguous or corrupted records. Candidate ranking and schema validation keep the output space controlled and auditable.

## Model Strategy

Primary model family:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Fine-tuning method:

```text
QLoRA, 4-bit NF4, LoRA rank 8, alpha 16, dropout 0.05
```

This size class is appropriate for OCR/log normalization because the task is constrained and schema-guided. Larger models can be evaluated later, but they should not be required for the first production-grade extraction layer.

## Training And Validation Contract

The delivery workflow must provide:

- dataset version and split contract;
- QLoRA adapter artifact checks;
- validation metrics;
- baseline comparison;
- simulation-ready event export;
- markdown and JSON delivery report.

The consolidated report is produced by:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_training_delivery python_executable=.venv/Scripts/python.exe
```

Outputs:

```text
reports\llm_training_delivery_report.json
reports\llm_training_delivery_report.md
reports\simulation_readiness.json
outputs\simulation_events.jsonl
```

## Acceptance Position

The LLM component can be accepted as a data-normalization and simulation-preparation module when the dataset contract, adapter artifact check, metric gate, baseline comparison, and simulation readiness gate pass.

Strategic poker policy approval remains a separate milestone.
