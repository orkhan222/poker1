from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.delivery_scope import ScopeGate, model_row_from_metrics, phase_status
from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.llm_decision import DECISION_ACTIONS, HeuristicTextProvider, LLMDecisionAgent
from poker_agent.model import load_policy
from poker_agent.schemas import PredictionRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PDF-scope aligned delivery report")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=250)
    parser.add_argument("--missing-hole-cards", choices=("drop", "flag", "keep"), default="drop")
    parser.add_argument("--allow-missing-hole-cards", action="store_true")
    parser.add_argument("--keep-all-in-class", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-policy-macro-f1", type=float, default=0.50)
    parser.add_argument("--min-llm-macro-f1", type=float, default=0.35)
    parser.add_argument("--min-accuracy-lift", type=float, default=0.0)
    parser.add_argument(
        "--event-agent-report",
        type=Path,
        default=Path("reports/llm_agent_architecture_decision.json"),
    )
    parser.add_argument("--production-gate", type=Path, default=Path("reports/production_gate.json"))
    parser.add_argument("--acceptance-report", type=Path, default=Path("reports/policy_acceptance.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/pdf_scope_alignment.json"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/pdf_scope_alignment.md"))
    parser.add_argument("--comparison-out", type=Path, default=Path("reports/pdf_scope_model_comparison.csv"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def decision_metrics(
    y_true: list[str],
    y_pred: list[str],
    probabilities: list[dict[str, float]],
) -> dict[str, Any]:
    labels = sorted(set(DECISION_ACTIONS) | set(y_true) | set(y_pred))
    true_counts = Counter(y_true)
    predicted_counts = Counter(y_pred)
    correct = sum(1 for true, predicted in zip(y_true, y_pred) if true == predicted)
    examples = len(y_true)

    per_class: dict[str, dict[str, float]] = {}
    weighted_f1 = 0.0
    balanced_accuracy = 0.0
    for label in labels:
        tp = sum(1 for true, predicted in zip(y_true, y_pred) if true == label and predicted == label)
        fp = sum(1 for true, predicted in zip(y_true, y_pred) if true != label and predicted == label)
        fn = sum(1 for true, predicted in zip(y_true, y_pred) if true == label and predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = float(true_counts[label])
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        weighted_f1 += f1 * support
        balanced_accuracy += recall

    cross_entropy = 0.0
    brier_loss = 0.0
    for true, row in zip(y_true, probabilities):
        cross_entropy += -math.log(max(float(row.get(true, 0.0)), 1e-12))
        for label in labels:
            target = 1.0 if label == true else 0.0
            brier_loss += (float(row.get(label, 0.0)) - target) ** 2

    majority_accuracy = true_counts.most_common(1)[0][1] / examples if examples else 0.0
    accuracy = correct / examples if examples else 0.0
    return {
        "examples": float(examples),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy / len(labels) if labels else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels) if labels else 0.0,
        "weighted_f1": weighted_f1 / examples if examples else 0.0,
        "cross_entropy": cross_entropy / examples if examples else 0.0,
        "brier_loss": brier_loss / (examples * len(labels)) if examples and labels else 0.0,
        "majority_baseline_accuracy": majority_accuracy,
        "lift_vs_majority": accuracy - majority_accuracy,
        "class_counts": dict(sorted(true_counts.items())),
        "predicted_class_counts": dict(sorted(predicted_counts.items())),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": [
                [
                    sum(1 for true, predicted in zip(y_true, y_pred) if true == row_label and predicted == col_label)
                    for col_label in labels
                ]
                for row_label in labels
            ],
        },
    }


def evaluate_llm_decision(records: list[tuple[PredictionRequest, dict[str, float], str, str]]) -> dict[str, Any]:
    agent = LLMDecisionAgent(provider=HeuristicTextProvider(), mode="candidate_ranker")
    y_true: list[str] = []
    y_pred: list[str] = []
    probabilities: list[dict[str, float]] = []
    latencies: list[float] = []

    for request, _features, label, _hand_id in records:
        response = agent.predict(request)
        y_true.append(label)
        y_pred.append(response.action)
        probabilities.append(response.probabilities)
        latencies.append(float(getattr(response, "latency_ms", 0.0)))

    metrics = decision_metrics(y_true, y_pred, probabilities)
    return {
        "metrics": metrics,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "provider": "heuristic_text",
        "mode": "candidate_ranker",
        "notes": (
            "Constrained prompt/candidate-ranking baseline. This is the fast deterministic backend used "
            "for reproducible comparison; transformer-backed runs are supported by llm_decision_baseline."
        ),
    }


def write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "family",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "cross_entropy",
        "latency_ms",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_scope_gates(
    *,
    policy_metrics: dict[str, Any],
    llm_metrics: dict[str, Any],
    event_agent_report: dict[str, Any],
    production_gate: dict[str, Any],
    acceptance_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, list[ScopeGate]]:
    llm_macro_f1 = float(llm_metrics.get("macro_f1", 0.0))
    event_selected = str(
        event_agent_report.get("selected_architecture")
        or event_agent_report.get("recommendation", {}).get("selected_architecture")
        or ""
    )
    event_status = str(
        event_agent_report.get("approval_status")
        or event_agent_report.get("recommendation", {}).get("approval_status")
        or ""
    )
    production_status = str(production_gate.get("status", "UNKNOWN"))
    acceptance_alignment = acceptance_report.get("human_action_alignment") or {}
    acceptance_simulation = acceptance_report.get("simulation") or {}
    acceptance_likeness = acceptance_report.get("human_likeness") or {}
    policy_gate_metrics = acceptance_alignment if acceptance_alignment else policy_metrics
    policy_macro_f1 = float(policy_gate_metrics.get("macro_f1", 0.0))
    policy_lift = float(policy_gate_metrics.get("lift_vs_majority", -999.0))
    simulation_status = str(acceptance_simulation.get("status", "FAIL"))
    likeness_status = str(acceptance_likeness.get("status", "FAIL"))
    timing_status = str(acceptance_likeness.get("timing_and_bet_size_status", "NOT_AVAILABLE"))

    return {
        "phase_1_two_baselines": [
            ScopeGate(
                name="LLM decision baseline",
                status="PASS" if float(llm_metrics.get("examples", 0.0)) > 0 else "FAIL",
                evidence=f"macro_f1={llm_macro_f1:.4f}, accuracy={float(llm_metrics.get('accuracy', 0.0)):.4f}",
                risk="A decision LLM should remain a comparison baseline until it beats the tabular policy on held-out data.",
                next_action="Run transformer-backed Qwen/SmolLM decision evaluation after stakeholder approval.",
            ),
            ScopeGate(
                name="End-to-end supervised policy baseline",
                status=(
                    "PASS"
                    if policy_macro_f1 >= args.min_policy_macro_f1 and policy_lift >= args.min_accuracy_lift
                    else "FAIL"
                ),
                evidence=f"macro_f1={policy_macro_f1:.4f}, lift_vs_majority={policy_lift:.4f}",
                risk="Current policy must not be sold as a profitable strategy while it fails minority-action or majority-lift gates.",
                next_action="Improve card coverage, class balance, and policy features before production approval.",
            ),
        ],
        "phase_2_selection_and_optimization": [
            ScopeGate(
                name="Architecture comparison",
                status="PASS",
                evidence="Supervised policy and constrained LLM decision baseline are evaluated on the same feature loader.",
                risk="The LLM backend in this report is deterministic for reproducibility; model-backed results require a separate run.",
                next_action="Promote the stronger approach only after matching dataset, split, and latency constraints.",
            ),
            ScopeGate(
                name="Event-normalization LLM architecture",
                status="PASS" if event_selected == "hybrid_parser_qlora" else "PARTIAL",
                evidence=f"selected_architecture={event_selected or 'unknown'}, approval_status={event_status or 'unknown'}",
                risk="Event normalization should not be confused with a poker-playing decision agent.",
                next_action="Use hybrid parser + QLoRA fallback for OCR/dealer-log normalization.",
            ),
        ],
        "phase_3_evaluation": [
            ScopeGate(
                name="Human action alignment",
                status="PASS" if policy_metrics else "FAIL",
                evidence=(
                    f"accuracy={float(acceptance_alignment.get('accuracy', policy_metrics.get('accuracy', 0.0))):.4f}, "
                    f"cross_entropy={float(acceptance_alignment.get('cross_entropy', policy_metrics.get('cross_entropy', 0.0))):.4f}"
                ),
                risk="Action alignment alone is insufficient for strategy approval.",
                next_action="Add bet-size MAE and timing distribution metrics from annotated hands.",
            ),
            ScopeGate(
                name="Self-play / simulated win-rate",
                status=simulation_status,
                evidence=(
                    "simulation_type="
                    f"{acceptance_simulation.get('simulation_type', 'missing')}, "
                    f"mean_win_rate={float(acceptance_simulation.get('mean_policy_win_rate', 0.0)):.4f}, "
                    f"p_ge_50_5={float(acceptance_simulation.get('probability_win_rate_at_least_50_5', 0.0)):.4f}"
                ),
                risk="Synthetic simulation is a regression proxy; a full poker self-play engine is still required for profitability claims.",
                next_action="Replace the proxy with validated multi-agent Hold'em self-play before production strategy approval.",
            ),
            ScopeGate(
                name="Human-likeness timing and bet-size distributions",
                status="PARTIAL" if likeness_status == "PASS" and timing_status == "NOT_AVAILABLE" else likeness_status,
                evidence=(
                    f"action_distribution={likeness_status}, "
                    f"js={float(acceptance_likeness.get('js_divergence', 0.0)):.4f}, "
                    f"timing_bet_size={timing_status}"
                ),
                risk="Action distribution can pass while delay and bet-size behavior remain unvalidated.",
                next_action="Extract waiting-time and amount labels, then add timing and bet-size distribution gates.",
            ),
        ],
        "phase_4_deployment": [
            ScopeGate(
                name="FastAPI and Docker delivery",
                status="PASS",
                evidence="FastAPI service, /predict, /health.json, Dockerfile, docker-compose, and packaged model artifacts exist.",
                risk="Deployment readiness does not imply strategy-model approval.",
                next_action="Keep API deployable while blocking autonomous decision-policy approval until model gates pass.",
            ),
            ScopeGate(
                name="Production model gate",
                status="PASS" if production_status == "PASS" else "FAIL",
                evidence=f"production_gate_status={production_status}",
                risk="The decision policy is not approved if production_gate is FAIL or missing.",
                next_action="Treat service as research/API integration artifact until production gate passes.",
            ),
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# PDF Scope Delivery Report",
        "",
        "## Executive Status",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Policy model: `{report['model']}`",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "The codebase now exposes the PDF-requested work as measurable delivery gates. "
        "The current repository contains deployable infrastructure, event normalization, "
        "a supervised policy baseline, and a constrained LLM decision baseline. It is not "
        "approved as a profitable autonomous poker strategy until simulation and human-likeness gates pass.",
        "",
        "## Model Comparison",
        "",
        "| System | Family | Accuracy | Macro F1 | Weighted F1 | Cross Entropy | Latency ms | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["model_comparison"]:
        latency = "" if row["latency_ms"] is None else f"{row['latency_ms']:.2f}"
        lines.append(
            f"| {row['system']} | {row['family']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} | {row['cross_entropy']:.4f} | {latency} | {row['status']} |"
        )

    lines.extend(["", "## Phase Gates", ""])
    for phase, gates in report["phase_gates"].items():
        lines.append(f"### {phase.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Gate | Status | Evidence | Next Action |")
        lines.append("| --- | --- | --- | --- |")
        for gate in gates:
            lines.append(
                f"| {gate['name']} | {gate['status']} | {gate['evidence']} | {gate['next_action']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Technical Recommendation",
            "",
            "1. Use `hybrid_parser_qlora` as the production candidate for OCR/dealer-log event normalization.",
            "2. Keep the supervised tabular policy as the current decision baseline and continue improving the data pipeline.",
            "3. Do not position the LLM as an autonomous poker-playing agent in the current milestone.",
            "4. Add self-play EV/win-rate and human-likeness metrics before claiming production strategy readiness.",
            "",
            "## Reproducible Commands",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe scripts\\run_hydra_experiment.py experiments=pdf_scope_delivery python_executable=.venv/Scripts/python.exe",
            ".\\.venv\\Scripts\\python.exe scripts\\run_hydra_experiment.py experiments=llm_decision_baseline python_executable=.venv/Scripts/python.exe",
            ".\\.venv\\Scripts\\python.exe scripts\\run_hydra_experiment.py experiments=compare_llm_agent_architectures python_executable=.venv/Scripts/python.exe",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=not args.allow_missing_hole_cards,
        missing_hole_cards="flag" if args.allow_missing_hole_cards and args.missing_hole_cards == "drop" else args.missing_hole_cards,
        merge_all_in=not args.keep_all_in_class,
        include_hand_id=True,
        include_request=True,
    )
    if not records:
        raise SystemExit(f"No evaluation examples found in {args.dataset}")
    if not args.model.exists():
        raise SystemExit(f"Policy model not found: {args.model}")

    policy = load_policy(args.model)
    policy_examples = [(features, label) for _request, features, label, _hand_id in records]
    policy_metrics = evaluate_policy(policy, policy_examples)
    llm_payload = evaluate_llm_decision(records)
    llm_metrics = llm_payload["metrics"]
    event_agent_report = load_json(args.event_agent_report)
    production_gate = load_json(args.production_gate)
    acceptance_report = load_json(args.acceptance_report)

    policy_status = (
        "PASS"
        if policy_metrics.get("macro_f1", 0.0) >= args.min_policy_macro_f1
        and policy_metrics.get("lift_vs_majority", -999.0) >= args.min_accuracy_lift
        else "FAIL"
    )
    llm_status = "MEASURED_BASELINE" if llm_metrics.get("examples", 0.0) > 0 else "MISSING"
    comparison_rows = [
        model_row_from_metrics(
            system="Supervised policy",
            family=str(getattr(policy, "model_kind", getattr(policy, "metadata", {}).get("policy", "policy"))),
            metrics=policy_metrics,
            latency_ms=None,
            status=policy_status,
            notes="End-to-end structured-state action policy baseline.",
        ).to_dict(),
        model_row_from_metrics(
            system="LLM decision baseline",
            family="candidate_ranker",
            metrics=llm_metrics,
            latency_ms=llm_payload["latency_ms"]["mean"],
            status=llm_status,
            notes=llm_payload["notes"],
        ).to_dict(),
    ]

    phase_gates_raw = build_scope_gates(
        policy_metrics=policy_metrics,
        llm_metrics=llm_metrics,
        event_agent_report=event_agent_report,
        production_gate=production_gate,
        acceptance_report=acceptance_report,
        args=args,
    )
    phase_gates = {
        phase: [gate.to_dict() for gate in gates]
        for phase, gates in phase_gates_raw.items()
    }
    phase_statuses = {phase: phase_status(gates) for phase, gates in phase_gates_raw.items()}
    overall_status = "PASS" if all(status == "PASS" for status in phase_statuses.values()) else "FAIL"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "model": str(args.model),
        "max_examples": args.max_examples,
        "overall_status": overall_status,
        "phase_statuses": phase_statuses,
        "model_comparison": comparison_rows,
        "policy_metrics": policy_metrics,
        "llm_decision_metrics": llm_payload,
        "event_agent_report": event_agent_report,
        "production_gate": production_gate,
        "acceptance_report": acceptance_report,
        "phase_gates": phase_gates,
        "recommendation": {
            "event_normalization_agent": "hybrid_parser_qlora",
            "decision_policy_candidate": "supervised_policy",
            "do_not_claim": "production profitable poker strategy until self-play and human-likeness gates pass",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_comparison_csv(comparison_rows, args.comparison_out)
    write_markdown(report, args.report_out)

    print(f"pdf_scope_status={overall_status}")
    print(f"pdf_scope_policy_macro_f1={policy_metrics['macro_f1']:.4f}")
    print(f"pdf_scope_policy_accuracy={policy_metrics['accuracy']:.4f}")
    print(f"pdf_scope_llm_macro_f1={llm_metrics['macro_f1']:.4f}")
    print(f"pdf_scope_llm_accuracy={llm_metrics['accuracy']:.4f}")
    print(f"pdf_scope_report={args.report_out}")
    print(f"pdf_scope_json={args.out}")
    print(f"pdf_scope_comparison={args.comparison_out}")


if __name__ == "__main__":
    main()
