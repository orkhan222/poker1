# Poker Decision Agent

Poker Decision Agent is a FastAPI service and ML research workspace for poker action prediction from OCR and event-log data. The repository includes the API, trained model artifact, Hydra experiment configs, evaluation scripts, audit reports, and a packaged delivery ZIP.

## Delivery Status

As of the latest delivery build:

```text
repository_audit=PASS
repo_hygiene=PASS
delivery_verification=PASS
model_production_gate=FAIL
```

The package is reproducible and ready for technical handoff. The model is not marked as production-approved for autonomous decision policy use, because the current dataset still has known coverage and class-balance limitations. Those limitations are documented in `reports\dataset_audit.json` and `reports\production_gate.json`.

## Repository Layout

```text
.
|-- poker_agent/              API, schemas, feature extraction, model loading
|-- scripts/                  training, evaluation, audit, packaging checks
|-- configs/                  Hydra experiment configuration
|-- docs/                     architecture and delivery decision records
|-- evaluation/               reviewed evaluation fixtures
|-- reports/                  generated metrics and audit outputs
|-- models/                   packaged model artifact
|-- release/                  delivery ZIP
|-- install.ps1               local environment setup
|-- run_server.ps1            API startup script
|-- complete_delivery.ps1     full delivery rebuild
|-- verify_delivery.ps1       final delivery verification
`-- README.md
```

## Install

```powershell
cd "C:\Users\user\Desktop\Secop\files-mentioned-by-the-user-poker-2"
.\install.ps1
```

## Run The API

```powershell
.\run_server.ps1
```

Open these endpoints after the server starts:

```text
http://127.0.0.1:8001/predict
http://127.0.0.1:8001/docs
http://127.0.0.1:8001/health.json
```

The health endpoint returns model status, policy name, split strategy, and the validation macro F1 stored in the model metadata.

## Architecture Notes

The LLM component is documented as a bounded event-normalization layer, not as
an autonomous poker decision system:

```text
docs\llm_agent_architecture.md
reports\llm_agent_architecture_decision.md
```

The recommended path is a schema-routed hybrid: deterministic parsing for
stable event families, LLM fallback only for ambiguous OCR/dealer-log text,
candidate ranking instead of free-form generation, and schema validation before
events are accepted into downstream datasets.

## Reproducible Experiments

Experiments are managed through Hydra. Each experiment has its own YAML file under `configs\experiments` and writes resolved configs, logs, and run metadata under `reports\hydra`.

Run any configured experiment with:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=<name> python_executable=.venv/Scripts/python.exe
```

Available experiment names:

```text
build_dataset
build_event_schema_dataset
phase2_event_benchmark
repo_hygiene
repo_audit
audit_dataset
train_single_hgb
evaluate_policy
research_compare_tabular
production_gate
train_routed_bundle_smoke
llm_event_extraction_smoke
llm_event_benchmark
llm_event_gold_eval
llm_transformer_gold_eval
llm_decision_baseline
compare_llm_agent_architectures
evaluate_hybrid_production
verify_delivery
```

Useful commands:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=repo_audit python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=build_event_schema_dataset python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=phase2_event_benchmark python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_event_benchmark python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_event_gold_eval python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_transformer_gold_eval python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_decision_baseline python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=compare_llm_agent_architectures python_executable=.venv/Scripts/python.exe
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=verify_delivery python_executable=.venv/Scripts/python.exe
```

Example override:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=train_single_hgb training.max_examples=5000 model.max_iter=40
```

Hydra output structure:

```text
reports\hydra\<experiment-name>\<timestamp>\
|-- environment.json
|-- artifact_manifest.json
|-- artifacts\
|-- resolved_config.yaml
|-- command.txt
|-- stdout.txt
|-- stderr.txt
`-- run.json
```

`environment.json` records the Python runtime, selected dependency versions,
git revision, dirty-state paths, seed, and thread settings. Output files are
hashed in `artifact_manifest.json` and copied into the run-local `artifacts`
directory when they are below the configured size limit. The repository audit
also verifies that every Hydra YAML declares every CLI argument supported by
its entrypoint and rejects CLI fallback defaults that are not owned by a Hydra
experiment configuration.

## Phase 1 Event Schema Dataset

The event extraction data layer includes a versioned schema, deterministic OCR
corruption expansion, and grouped train/validation/test splits.

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=build_event_schema_dataset python_executable=.venv/Scripts/python.exe
```

Generated files:

```text
evaluation\event_schema_v1.json
evaluation\event_extraction_phase1.jsonl
evaluation\event_extraction_phase1_splits.json
reports\event_schema_dataset_report.json
reports\event_schema_dataset_report.md
```

Latest Phase 1 dataset summary:

```text
schema_version=event_schema_v1
parent_examples=24
expanded_examples=120
groups=24
train_rows=75
valid_rows=20
test_rows=25
validation_status=PASS
```

## Phase 2 Baseline Benchmark

The Phase 2 benchmark compares deterministic parsing, zero-shot extraction,
few-shot extraction with 5 and 10 examples, and candidate ranking on the same
grouped event-normalization test split.

Default reproducible local run:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=phase2_event_benchmark python_executable=.venv/Scripts/python.exe
```

Generated files:

```text
reports\phase2_event_benchmark_results.csv
reports\phase2_event_benchmark_results.json
reports\phase2_event_benchmark_predictions.jsonl
reports\phase2_event_benchmark_report.md
```

Latest default results on the Phase 1 test split:

```text
examples=25
best_method=deterministic_parser
best_accuracy=1.0000
best_macro_f1=1.0000
schema_validity_rate=1.0000
```

The default run uses a deterministic local backend so delivery checks remain
fast and reproducible. To run the same benchmark against local or downloadable
instruction models, switch the backend and model IDs:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=phase2_event_benchmark experiments.command.args.backend=transformers "experiments.command.args.model_ids=qwen2_5_1_5b,qwen3_1_7b,smollm2_1_7b" python_executable=.venv/Scripts/python.exe
```

## LLM Agent Architecture Decision

The production LLM agent is implemented as a parser-first event-normalization
agent with lazy QLoRA fallback. The trained Qwen2.5 QLoRA adapter is retained
as a model artifact and research baseline, but direct free-form extraction is
not selected for production because it does not pass the schema and simulation
readiness gates.

Run the architecture comparison:

```powershell
.\.venv\Scripts\python.exe scripts\compare_llm_agent_architectures.py --config config.yaml
```

Run the same comparison through Hydra:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=compare_llm_agent_architectures python_executable=.venv/Scripts/python.exe
```

Generated files:

```text
reports\llm_agent_architecture_comparison.csv
reports\llm_agent_architecture_decision.json
reports\llm_agent_architecture_decision.md
outputs\llm_agent_architecture_predictions.jsonl
```

Current proposed architecture:

```text
selected_architecture=hybrid_parser_qlora
approval_status=proposed_for_stakeholder_approval
```

## Text Event Extraction Results

The repository includes a text/event extraction benchmark for turning OCR and dealer-log records into structured poker events. This is used to improve betting-history reconstruction before model training.

Weak-label benchmark on 1000 log records:

```text
value_only_baseline: event_accuracy=0.4150, macro_f1=0.3284
local_rules:         event_accuracy=1.0000, macro_f1=1.0000
```

Gold-label evaluation on 24 reviewed examples:

```text
minimal_action_only:      event_accuracy=0.6667, macro_f1=0.4091
permissive_prompt_rules:  event_accuracy=0.8333, macro_f1=0.8545
strict_schema_rules:      event_accuracy=1.0000, macro_f1=1.0000
```

The strict schema approach is the strongest current extractor. Card extraction still needs more validation: `strict_schema_rules` reaches `card_exact_match=0.8000` on the current gold fixture. The next data-quality step is to expand the reviewed fixture and enforce rank/suit validation before extracted cards are used as supervised labels.


### Local Instruction Model Experiment

A real local instruction model experiment uses
`HuggingFaceTB/SmolLM2-135M-Instruct` on the same 24 reviewed examples with
deterministic CPU inference. The first run downloads the model from Hugging
Face.

```text
strict_zero_shot: event_accuracy=0.2917, macro_f1=0.1129
few_shot:         event_accuracy=0.3750, macro_f1=0.1364
candidate_ranker: event_accuracy=0.3750, macro_f1=0.1364
calibrated_ranker:event_accuracy=0.3750, macro_f1=0.1406
schema_routed_hybrid: event_accuracy=1.0000, macro_f1=1.0000
```

Few-shot examples improved event accuracy by `0.0833` and macro F1 by `0.0235`,
while contextual calibration improved candidate-ranking macro F1 by `0.0043`.
The production-oriented schema-routed hybrid reached `1.0000` accuracy and
macro F1 by validating known structured event families before invoking the
zero-shot model for other event types. Router coverage was `0.9167`; the real
LLM fallback processed `2/24` examples (`0.0833`) with `1.0000` fallback
accuracy. This result must be revalidated on a larger fixture with ambiguous
and corrupted event names.

## LLM Decision Baseline

The repository now includes a constrained text-model decision baseline for
Phase 1 comparison against the supervised policy model. It converts structured
game state into a compact decision prompt, parses the output into one action,
and reports accuracy, macro F1, cross-entropy, invalid output rate, latency,
and confusion matrix.

Default reproducible run:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_decision_baseline python_executable=.venv/Scripts/python.exe
```

Direct run:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_llm_decision_agent.py --dataset "C:\Users\user\Desktop\AllFile\dataset" --max-examples 250
```

The default config uses a deterministic local scoring provider so the baseline
is fast and reproducible. For a real local model run, override the provider and
model settings:

```powershell
.\.venv\Scripts\python.exe scripts\run_hydra_experiment.py experiments=llm_decision_baseline model.provider=transformers model.model_id=Qwen/Qwen2.5-1.5B-Instruct training.max_examples=100
```

## Latest Model Metrics

Current packaged policy:

```text
policy=hist_gradient_boosting
split=stratified_hand_group_holdout
valid_accuracy=0.6798
valid_balanced_accuracy=0.4415
valid_macro_f1=0.4135
valid_weighted_f1=0.6636
valid_majority_baseline_accuracy=0.7029
valid_lift_vs_majority=-0.0231
```

The model is suitable for API integration, data-pipeline testing, and research iteration. It should not be presented as a completed profitable strategy model until the production gate passes.

## Key Reports

```text
reports\repository_audit.json
reports\repo_hygiene.json
reports\dataset_audit.json
reports\production_gate.json
reports\event_schema_dataset_report.json
reports\event_schema_dataset_report.md
reports\phase2_event_benchmark_results.csv
reports\phase2_event_benchmark_results.json
reports\phase2_event_benchmark_report.md
reports\llm_event_benchmark.json
reports\llm_event_gold_eval.json
reports\llm_event_gold_report.md
reports\llm_transformer_gold_eval.json
reports\llm_transformer_gold_report.md
reports\llm_decision_agent_eval.json
reports\llm_decision_agent_report.md
reports\delivery_verification.json
reports\delivery_report.md
```

## Build The Delivery Package

```powershell
.\complete_delivery.ps1 -SkipTrain -AllowGateFailure
```

Use `-SkipTrain` when rebuilding the delivery package around the existing model. Remove it when a fresh training run is required.

Final ZIP:

```text
release\poker-decision-agent.zip
```

## Verify The Delivery

```powershell
.\verify_delivery.ps1
```

Expected result:

```text
"status": "PASS"
```

## Open Risks

- Hole-card coverage is still too low for reliable card-strength modeling.
- The target distribution is imbalanced and fold-dominant.
- The current model does not beat the majority-class baseline on strict holdout accuracy.
- The gold event extraction set is intentionally small and should be expanded with reviewed production logs.
