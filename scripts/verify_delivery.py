from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.agents import MLPolicyAgent
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest
from poker_agent.service import health_payload, resolve_model_path


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the poker agent delivery package")
    parser.add_argument("--project-root", default=ROOT, type=Path)
    parser.add_argument("--model", default=ROOT / "models" / "poker_policy.joblib", type=Path)
    parser.add_argument("--zip", default=ROOT / "release" / "poker-decision-agent.zip", type=Path)
    parser.add_argument("--require-gate-pass", action="store_true")
    parser.add_argument("--json-out", default=None, type=Path)
    return parser.parse_args()


def run_check(name: str, fn: Callable[[], str]) -> Check:
    try:
        return Check(name=name, passed=True, detail=fn())
    except Exception as exc:
        return Check(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}")


def require_files(root: Path) -> str:
    required = [
        "README.md",
        "docs/llm_agent_architecture.md",
        "evaluation.ipynb",
        "requirements.txt",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/model/tabular_compare.yaml",
        "configs/model/routed_bundle_smoke.yaml",
        "configs/model/text_event_local_rules.yaml",
        "configs/model/text_event_smol.yaml",
        "configs/model/text_decision_local.yaml",
        "configs/training/group_holdout.yaml",
        "configs/training/smoke.yaml",
        "configs/evaluation/standard.yaml",
        "configs/inference/local_service.yaml",
        "configs/logging/local.yaml",
        "configs/prompts/event_extraction_prompt.txt",
        "configs/prompts/event_extraction_minimal.txt",
        "configs/prompts/event_extraction_permissive.txt",
        "configs/prompts/event_extraction_strict.txt",
        "configs/prompts/event_extraction_fewshot.txt",
        "configs/prompts/event_type_candidate_ranker.txt",
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/build_event_schema_dataset.yaml",
        "configs/experiments/phase2_event_benchmark.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/evaluate_policy.yaml",
        "configs/experiments/research_compare_tabular.yaml",
        "configs/experiments/audit_dataset.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/production_gate.yaml",
        "configs/experiments/train_routed_bundle_smoke.yaml",
        "configs/experiments/llm_event_extraction_smoke.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "configs/experiments/llm_transformer_gold_eval.yaml",
        "configs/experiments/llm_decision_baseline.yaml",
        "configs/experiments/verify_delivery.yaml",
        "Dockerfile",
        "docker-compose.yml",
        "install.ps1",
        "run_server.ps1",
        "complete_delivery.ps1",
        "verify_delivery.ps1",
        "models/poker_policy.joblib",
        "reports/dataset_audit.json",
        "reports/repository_audit.json",
        "reports/production_gate.json",
        "reports/event_schema_dataset_report.json",
        "reports/event_schema_dataset_report.md",
        "reports/phase2_event_benchmark_results.csv",
        "reports/phase2_event_benchmark_results.json",
        "reports/phase2_event_benchmark_predictions.jsonl",
        "reports/phase2_event_benchmark_report.md",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_report.md",
        "reports/llm_event_methodology.md",
        "reports/llm_transformer_gold_eval.json",
        "reports/llm_transformer_gold_report.md",
        "reports/llm_decision_agent_eval.json",
        "reports/llm_decision_agent_report.md",
        "reports/delivery_report.md",
        "evaluation/event_extraction_gold.jsonl",
        "evaluation/event_schema_v1.json",
        "evaluation/event_extraction_phase1.jsonl",
        "evaluation/event_extraction_phase1_splits.json",
        "scripts/benchmark.py",
        "scripts/build_event_schema_dataset.py",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/evaluate_policy.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/check_repo_hygiene.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_event_extraction.py",
        "scripts/llm_transformer_gold_eval.py",
        "scripts/evaluate_llm_decision_agent.py",
        "scripts/production_gate.py",
        "scripts/run_hydra_experiment.py",
        "scripts/verify_delivery.py",
        "poker_agent/service.py",
        "poker_agent/agents.py",
        "poker_agent/features.py",
        "poker_agent/event_schema.py",
        "poker_agent/event_normalization/__init__.py",
        "poker_agent/event_normalization/backends.py",
        "poker_agent/event_normalization/benchmark.py",
        "poker_agent/event_normalization/candidate_ranker.py",
        "poker_agent/event_normalization/few_shot.py",
        "poker_agent/event_normalization/metrics.py",
        "poker_agent/event_normalization/parser.py",
        "poker_agent/event_normalization/prompts.py",
        "poker_agent/event_normalization/schema.py",
        "poker_agent/event_normalization/zero_shot.py",
        "poker_agent/llm_decision.py",
        "poker_agent/model.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
    ]
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        raise AssertionError(f"Missing required files: {missing}")
    return f"{len(required)} required files present"


def compile_sources(root: Path) -> str:
    source_files = [
        "poker_agent/agents.py",
        "poker_agent/evaluator.py",
        "poker_agent/event_schema.py",
        "poker_agent/event_normalization/__init__.py",
        "poker_agent/event_normalization/backends.py",
        "poker_agent/event_normalization/benchmark.py",
        "poker_agent/event_normalization/candidate_ranker.py",
        "poker_agent/event_normalization/few_shot.py",
        "poker_agent/event_normalization/metrics.py",
        "poker_agent/event_normalization/parser.py",
        "poker_agent/event_normalization/prompts.py",
        "poker_agent/event_normalization/schema.py",
        "poker_agent/event_normalization/zero_shot.py",
        "poker_agent/features.py",
        "poker_agent/llm_decision.py",
        "poker_agent/model.py",
        "poker_agent/schemas.py",
        "poker_agent/service.py",
        "poker_agent/slices.py",
        "poker_agent/validation.py",
        "scripts/audit_dataset.py",
        "scripts/audit_repository.py",
        "scripts/check_repo_hygiene.py",
        "scripts/benchmark.py",
        "scripts/build_event_schema_dataset.py",
        "scripts/evaluate_policy.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_event_extraction.py",
        "scripts/llm_transformer_gold_eval.py",
        "scripts/evaluate_llm_decision_agent.py",
        "scripts/production_gate.py",
        "scripts/research_experiment.py",
        "scripts/run_hydra_experiment.py",
        "scripts/train_policy.py",
        "scripts/train_policy_bundle.py",
        "scripts/verify_delivery.py",
    ]
    for relative in source_files:
        path = root / relative
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return f"{len(source_files)} Python files compile without writing bytecode"


def model_loads(model_path: Path) -> str:
    model = load_policy(model_path)
    metadata = getattr(model, "metadata", {}) or {}
    if not metadata:
        raise AssertionError("Model artifact has no metadata")
    split = (metadata.get("split") or {}).get("split_type")
    if split != "stratified_hand_group_holdout":
        raise AssertionError(f"Unexpected split: {split}")
    valid = metadata.get("valid_metrics") or {}
    if "macro_f1" not in valid:
        raise AssertionError("Model metadata does not include validation metrics")
    return f"model={model_path.name}, policy={metadata.get('policy')}, macro_f1={valid['macro_f1']:.4f}"


def inference_contract(model_path: Path) -> str:
    agent = MLPolicyAgent.from_path(model_path)
    observed = agent.predict(
        PredictionRequest(
            position="BTN",
            street="preflop",
            hole_cards=["Ah", "Kd"],
            board_cards=[],
            pot=2.5,
            to_call=1.0,
            stack=100.0,
            min_raise=2.0,
            player_count=6,
        )
    ).to_dict()
    missing = agent.predict(
        PredictionRequest(
            position="BTN",
            street="preflop",
            hole_cards=[],
            board_cards=[],
            pot=2.5,
            to_call=1.0,
            stack=100.0,
            min_raise=2.0,
            player_count=6,
        )
    ).to_dict()
    if observed["model_status"] == "missing_card_fallback":
        raise AssertionError("Observed-card request incorrectly used fallback")
    if missing["model_status"] != "missing_card_fallback":
        raise AssertionError("Missing-card request did not use fallback")
    for payload in (observed, missing):
        total = sum(float(value) for value in payload["probabilities"].values())
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(f"Probabilities do not sum to 1: {total}")
    return f"observed={observed['action']} missing={missing['action']}"


def health_contract(model_path: Path) -> str:
    resolved = resolve_model_path()
    if resolved.resolve() != model_path.resolve():
        raise AssertionError(f"Health resolved unexpected model path: {resolved}")
    payload = health_payload()
    if payload.get("model_status") != "loaded":
        raise AssertionError(f"Model status is not loaded: {payload}")
    if "valid_macro_f1" not in payload:
        raise AssertionError(f"Health payload missing model metric metadata: {payload}")
    return json.dumps(payload, sort_keys=True)


def reports_contract(root: Path, require_gate_pass: bool) -> str:
    audit = json.loads((root / "reports" / "dataset_audit.json").read_text(encoding="utf-8"))
    repo_audit = json.loads((root / "reports" / "repository_audit.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "reports" / "production_gate.json").read_text(encoding="utf-8"))
    event_schema_report = root / "reports" / "event_schema_dataset_report.json"
    phase2_report = root / "reports" / "phase2_event_benchmark_results.json"
    benchmark = root / "reports" / "llm_event_benchmark.json"
    gold_eval = root / "reports" / "llm_event_gold_eval.json"
    transformer_eval = root / "reports" / "llm_transformer_gold_eval.json"
    decision_eval = root / "reports" / "llm_decision_agent_eval.json"
    if "findings" not in audit:
        raise AssertionError("Audit report has no findings key")
    if repo_audit.get("status") != "PASS":
        raise AssertionError("Repository audit did not pass")
    hydra_audit = repo_audit.get("hydra", {})
    if hydra_audit.get("missing_hydra_configs"):
        raise AssertionError(f"Hydra configs are missing: {hydra_audit['missing_hydra_configs']}")
    if hydra_audit.get("incomplete_argument_configs"):
        raise AssertionError(f"Hydra argument coverage is incomplete: {hydra_audit['incomplete_argument_configs']}")
    if repo_audit.get("unowned_hardcoded_defaults"):
        raise AssertionError(f"CLI defaults are not owned by Hydra configs: {repo_audit['unowned_hardcoded_defaults']}")
    if gate.get("status") not in {"PASS", "FAIL"}:
        raise AssertionError(f"Invalid gate status: {gate.get('status')}")
    if require_gate_pass and gate.get("status") != "PASS":
        raise AssertionError("Production gate did not pass")
    if not event_schema_report.exists():
        raise AssertionError("Phase 1 event schema dataset report is missing")
    schema_payload = json.loads(event_schema_report.read_text(encoding="utf-8"))
    if schema_payload.get("status") != "PASS":
        raise AssertionError(f"Phase 1 event schema dataset did not pass validation: {schema_payload.get('status')}")
    if schema_payload.get("examples", 0) < 100:
        raise AssertionError("Phase 1 event schema dataset has fewer than 100 rows")
    if schema_payload.get("groups", 0) < 20:
        raise AssertionError("Phase 1 event schema dataset has insufficient grouped parents")
    if set(schema_payload.get("split_counts", {})) != {"train", "valid", "test"}:
        raise AssertionError("Phase 1 event schema dataset does not include train/valid/test splits")
    benchmark_detail = (
        f", event_schema_rows={schema_payload.get('examples')}"
        f", event_schema_groups={schema_payload.get('groups')}"
    )
    if not phase2_report.exists():
        raise AssertionError("Phase 2 event-normalization benchmark report is missing")
    phase2_payload = json.loads(phase2_report.read_text(encoding="utf-8"))
    phase2_systems = phase2_payload.get("systems", [])
    phase2_methods = {system.get("method") for system in phase2_systems}
    required_phase2_methods = {"deterministic_parser", "zero_shot", "few_shot_5", "few_shot_10", "candidate_ranker"}
    if not required_phase2_methods.issubset(phase2_methods):
        raise AssertionError(f"Phase 2 benchmark is missing methods: {sorted(required_phase2_methods - phase2_methods)}")
    if phase2_payload.get("examples", 0) <= 0:
        raise AssertionError("Phase 2 benchmark has no evaluated examples")
    best_phase2 = max(phase2_systems, key=lambda item: item.get("macro_f1", 0.0))
    if best_phase2.get("schema_validity_rate", 0.0) < 0.99:
        raise AssertionError("Phase 2 benchmark best system has insufficient schema validity")
    benchmark_detail += (
        f", phase2_examples={phase2_payload.get('examples')}"
        f", phase2_best={best_phase2.get('method')}"
        f", phase2_macro_f1={best_phase2.get('macro_f1')}"
    )
    if benchmark.exists():
        benchmark_payload = json.loads(benchmark.read_text(encoding="utf-8"))
        if "systems" not in benchmark_payload:
            raise AssertionError("Event extraction benchmark has no systems key")
        benchmark_detail += f", event_benchmark_records={benchmark_payload.get('records_evaluated')}"
    if not gold_eval.exists():
        raise AssertionError("Gold event extraction evaluation report is missing")
    gold_payload = json.loads(gold_eval.read_text(encoding="utf-8"))
    strict_metrics = gold_payload.get("systems", {}).get("strict_schema_rules", {})
    if strict_metrics.get("event_type", {}).get("macro_f1", 0.0) < 0.90:
        raise AssertionError("Gold event extraction macro F1 is below acceptance threshold")
    benchmark_detail += f", gold_examples={gold_payload.get('examples')}"
    if not transformer_eval.exists():
        raise AssertionError("Local instruction-model evaluation report is missing")
    transformer_payload = json.loads(transformer_eval.read_text(encoding="utf-8"))
    systems = transformer_payload.get("systems", {})
    zero_shot = systems.get("smol_strict_zero_shot", {}).get("event_type", {})
    few_shot = systems.get("smol_few_shot", {}).get("event_type", {})
    ranker = systems.get("smol_candidate_ranker", {}).get("event_type", {})
    calibrated = systems.get("smol_calibrated_candidate_ranker", {}).get("event_type", {})
    hybrid_metrics = systems.get("schema_routed_smol_hybrid", {})
    hybrid = hybrid_metrics.get("event_type", {})
    if not transformer_payload.get("model_id"):
        raise AssertionError("Instruction-model evaluation has no model id")
    if few_shot.get("accuracy", 0.0) < zero_shot.get("accuracy", 0.0):
        raise AssertionError("Few-shot prompt regressed against strict zero-shot accuracy")
    if calibrated.get("macro_f1", 0.0) < ranker.get("macro_f1", 0.0):
        raise AssertionError("Contextual calibration regressed against uncalibrated candidate ranking")
    if hybrid.get("macro_f1", 0.0) < 0.90:
        raise AssertionError("Schema-routed LLM hybrid macro F1 is below acceptance threshold")
    if hybrid_metrics.get("llm_fallback_count", 0) <= 0:
        raise AssertionError("Schema-routed hybrid did not exercise the LLM fallback")
    benchmark_detail += (
        f", transformer_model={transformer_payload.get('model_id')}"
        f", calibrated_macro_f1={calibrated.get('macro_f1')}"
        f", hybrid_macro_f1={hybrid.get('macro_f1')}"
        f", hybrid_llm_fallback_rate={hybrid_metrics.get('llm_fallback_rate')}"
    )
    if not decision_eval.exists():
        raise AssertionError("Text-model decision baseline report is missing")
    decision_payload = json.loads(decision_eval.read_text(encoding="utf-8"))
    decision_metrics = decision_payload.get("metrics", {})
    if decision_metrics.get("examples", 0.0) <= 0.0:
        raise AssertionError("Text-model decision baseline has no evaluated examples")
    if decision_payload.get("invalid_output_rate", 1.0) > 0.05:
        raise AssertionError("Text-model decision baseline invalid output rate is too high")
    benchmark_detail += (
        f", decision_examples={int(decision_metrics.get('examples', 0.0))}"
        f", decision_accuracy={decision_metrics.get('accuracy')}"
        f", decision_macro_f1={decision_metrics.get('macro_f1')}"
    )
    return (
        f"audit_findings={len(audit.get('findings', []))}, "
        f"repo_audit={repo_audit.get('status')}, gate={gate.get('status')}{benchmark_detail}"
    )


def repo_hygiene_contract(root: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "check_repo_hygiene.py"), "--root", str(root)],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip() or completed.stderr.strip()
        raise AssertionError(detail[:2000])
    payload = json.loads(completed.stdout)
    return f"hygiene={payload['status']}"


def hydra_provenance_contract(root: Path) -> str:
    experiment_root = root / "reports" / "hydra" / "llm_transformer_gold_eval"
    runs = sorted(path for path in experiment_root.glob("*") if path.is_dir())
    if not runs:
        raise AssertionError("No Hydra LLM experiment runs were found")
    latest = runs[-1]
    required = {
        "resolved_config.yaml",
        "command.txt",
        "stdout.txt",
        "stderr.txt",
        "run.json",
        "environment.json",
        "artifact_manifest.json",
    }
    missing = sorted(name for name in required if not (latest / name).exists())
    if missing:
        raise AssertionError(f"Hydra run provenance is incomplete: {missing}")
    run = json.loads((latest / "run.json").read_text(encoding="utf-8"))
    environment = json.loads((latest / "environment.json").read_text(encoding="utf-8"))
    artifacts = json.loads((latest / "artifact_manifest.json").read_text(encoding="utf-8"))
    if run.get("status") != "pass" or not run.get("deterministic"):
        raise AssertionError(f"Latest Hydra LLM run is not deterministic/pass: {run.get('status')}")
    if not environment.get("packages", {}).get("transformers"):
        raise AssertionError("Hydra environment manifest does not record transformers")
    file_artifacts = [item for item in artifacts.get("artifacts", []) if item.get("type") == "file"]
    if not file_artifacts or any(not item.get("sha256") for item in file_artifacts):
        raise AssertionError("Hydra artifact manifest is missing file checksums")
    return f"run={latest.name}, artifacts={len(file_artifacts)}, git={environment.get('git', {}).get('revision')}"


def zip_contract(root: Path, zip_path: Path) -> str:
    required = {
        "models/poker_policy.joblib",
        "README.md",
        "docs/llm_agent_architecture.md",
        "evaluation.ipynb",
        "configs/experiment.yaml",
        "configs/dataset/poker_csv.yaml",
        "configs/model/hist_gradient_boosting.yaml",
        "configs/model/text_event_smol.yaml",
        "configs/model/text_decision_local.yaml",
        "configs/prompts/event_type_candidate_ranker.txt",
        "configs/experiments/build_dataset.yaml",
        "configs/experiments/build_event_schema_dataset.yaml",
        "configs/experiments/phase2_event_benchmark.yaml",
        "configs/experiments/repo_hygiene.yaml",
        "configs/experiments/train_single_hgb.yaml",
        "configs/experiments/repo_audit.yaml",
        "configs/experiments/llm_event_benchmark.yaml",
        "configs/experiments/llm_event_gold_eval.yaml",
        "configs/experiments/llm_transformer_gold_eval.yaml",
        "configs/experiments/llm_decision_baseline.yaml",
        "evaluation/event_extraction_gold.jsonl",
        "evaluation/event_schema_v1.json",
        "evaluation/event_extraction_phase1.jsonl",
        "evaluation/event_extraction_phase1_splits.json",
        "reports/dataset_audit.json",
        "reports/repository_audit.json",
        "reports/production_gate.json",
        "reports/event_schema_dataset_report.json",
        "reports/event_schema_dataset_report.md",
        "reports/phase2_event_benchmark_results.csv",
        "reports/phase2_event_benchmark_results.json",
        "reports/phase2_event_benchmark_predictions.jsonl",
        "reports/phase2_event_benchmark_report.md",
        "reports/llm_event_benchmark.json",
        "reports/llm_event_gold_eval.json",
        "reports/llm_event_gold_report.md",
        "reports/llm_event_methodology.md",
        "reports/llm_transformer_gold_eval.json",
        "reports/llm_transformer_gold_report.md",
        "reports/llm_decision_agent_eval.json",
        "reports/llm_decision_agent_report.md",
        "reports/delivery_report.md",
        "scripts/benchmark.py",
        "scripts/check_repo_hygiene.py",
        "scripts/audit_repository.py",
        "scripts/build_event_schema_dataset.py",
        "scripts/llm_event_benchmark.py",
        "scripts/llm_event_gold_eval.py",
        "scripts/llm_transformer_gold_eval.py",
        "scripts/evaluate_llm_decision_agent.py",
        "scripts/run_hydra_experiment.py",
        "scripts/verify_delivery.py",
        "poker_agent/event_schema.py",
        "poker_agent/event_normalization/__init__.py",
        "poker_agent/event_normalization/backends.py",
        "poker_agent/event_normalization/benchmark.py",
        "poker_agent/event_normalization/candidate_ranker.py",
        "poker_agent/event_normalization/few_shot.py",
        "poker_agent/event_normalization/metrics.py",
        "poker_agent/event_normalization/parser.py",
        "poker_agent/event_normalization/prompts.py",
        "poker_agent/event_normalization/schema.py",
        "poker_agent/event_normalization/zero_shot.py",
        "verify_delivery.ps1",
    }
    if not zip_path.exists():
        raise AssertionError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    forbidden = sorted(
        name
        for name in names
        if "__pycache__/" in name
        or name.endswith((".pyc", ".pyo", ".pyd"))
        or name.endswith("requirements-research.txt")
    )
    if forbidden:
        raise AssertionError(f"ZIP contains generated or removed artifacts: {forbidden[:20]}")
    missing = sorted(required - names)
    if missing:
        raise AssertionError(f"ZIP is missing required entries: {missing}")
    if not any(name.endswith("/environment.json") for name in names):
        raise AssertionError("ZIP contains no Hydra environment manifest")
    if not any(name.endswith("/artifact_manifest.json") for name in names):
        raise AssertionError("ZIP contains no Hydra artifact manifest")
    return f"zip_entries={len(names)}"


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    checks = [
        run_check("required_files", lambda: require_files(root)),
        run_check("compile_sources", lambda: compile_sources(root)),
        run_check("model_loads", lambda: model_loads(args.model)),
        run_check("inference_contract", lambda: inference_contract(args.model)),
        run_check("health_contract", lambda: health_contract(args.model)),
        run_check("reports_contract", lambda: reports_contract(root, args.require_gate_pass)),
        run_check("repo_hygiene_contract", lambda: repo_hygiene_contract(root)),
        run_check("hydra_provenance_contract", lambda: hydra_provenance_contract(root)),
        run_check("zip_contract", lambda: zip_contract(root, args.zip)),
    ]
    payload = {
        "status": "PASS" if all(check.passed for check in checks) else "FAIL",
        "checks": [check.__dict__ for check in checks],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
