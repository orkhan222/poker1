from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


STRATEGY_READINESS_VERSION = "2026-06-19"


REMEDIATION_BY_GATE: dict[str, dict[str, str]] = {
    "accuracy_lift": {
        "severity": "critical",
        "owner": "modeling",
        "required_fix": "Train a challenger policy that beats the majority-class baseline on the same grouped validation split.",
    },
    "macro_f1": {
        "severity": "critical",
        "owner": "modeling",
        "required_fix": "Improve minority-action recall with stronger features, class-aware training, and model selection on macro F1.",
    },
    "balanced_accuracy": {
        "severity": "high",
        "owner": "modeling",
        "required_fix": "Raise per-class recall by reducing fold-class dominance through weighted loss, resampling, or routed policies.",
    },
    "observed_hole_cards_macro_f1": {
        "severity": "high",
        "owner": "data",
        "required_fix": "Improve card extraction coverage and validate card-strength features on hands where hole cards are present.",
    },
    "facing_bet_macro_f1": {
        "severity": "high",
        "owner": "modeling",
        "required_fix": "Add pressure-situation features and evaluate call/fold/raise behavior on facing-bet slices.",
    },
    "dataset_audit_blockers": {
        "severity": "critical",
        "owner": "data",
        "required_fix": "Resolve blocker-level dataset audit findings before approving the policy for production strategy use.",
    },
    "calibration": {
        "severity": "medium",
        "owner": "modeling",
        "required_fix": "Recalibrate probabilities with a validation-only calibration layer if confidence error exceeds threshold.",
    },
    "validation_split": {
        "severity": "critical",
        "owner": "evaluation",
        "required_fix": "Use grouped, session, or source holdout validation to prevent action-level leakage.",
    },
}


MINIMUM_EXIT_CRITERIA: dict[str, str] = {
    "validation": "Grouped validation split remains enabled and reproducible.",
    "baseline_lift": "Accuracy lift versus majority baseline is positive on the same split.",
    "macro_f1": "Validation macro F1 reaches the production threshold.",
    "balanced_accuracy": "Balanced accuracy reaches the production threshold.",
    "pressure_slice": "Facing-bet slice clears the configured macro-F1 threshold.",
    "dataset_blockers": "Dataset audit has no blocker-level findings.",
    "simulation": "Simulation and human-likeness gates pass before any autonomous strategic deployment.",
}


def summarize_strategy_readiness(report: dict[str, Any]) -> dict[str, Any]:
    gates = list(report.get("gates", []))
    valid_metrics = dict(report.get("valid_metrics", {}))
    failing_gates = [gate for gate in gates if not gate.get("passed")]
    blocking_reasons = [_blocking_reason(gate) for gate in failing_gates]
    approved = str(report.get("status")).upper() == "PASS" and not blocking_reasons

    return {
        "version": STRATEGY_READINESS_VERSION,
        "strategy_policy_status": "APPROVED" if approved else "NOT_APPROVED",
        "production_gate_status": "PASS" if approved else "FAIL",
        "deployment_mode": "production_policy" if approved else "technical_handoff_only",
        "decision": (
            "Approved for production decision-policy deployment."
            if approved
            else "Blocked from production decision-policy deployment."
        ),
        "approval_boundary": {
            "software_delivery": "Service, API contract, reports, packaging, and reproducibility checks can pass independently.",
            "strategy_policy": "Strategic approval requires model-quality, dataset-quality, and simulation gates to pass.",
        },
        "metric_snapshot": {
            "accuracy": valid_metrics.get("accuracy"),
            "majority_baseline_accuracy": valid_metrics.get("majority_baseline_accuracy"),
            "accuracy_lift": valid_metrics.get("lift_vs_majority"),
            "macro_f1": valid_metrics.get("macro_f1"),
            "balanced_accuracy": valid_metrics.get("balanced_accuracy"),
            "weighted_f1": valid_metrics.get("weighted_f1"),
            "ece_10": valid_metrics.get("ece_10"),
        },
        "blocking_reasons": blocking_reasons,
        "minimum_exit_criteria": deepcopy(MINIMUM_EXIT_CRITERIA),
        "recommended_next_milestone": _next_milestone(blocking_reasons),
    }


def load_strategy_readiness(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {
            "version": STRATEGY_READINESS_VERSION,
            "strategy_policy_status": "UNKNOWN",
            "production_gate_status": "MISSING_REPORT",
            "deployment_mode": "unavailable",
            "decision": f"Production gate report was not found: {report_path}",
            "blocking_reasons": [],
            "minimum_exit_criteria": deepcopy(MINIMUM_EXIT_CRITERIA),
        }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("strategy_readiness") or summarize_strategy_readiness(report)


def _blocking_reason(gate: dict[str, Any]) -> dict[str, Any]:
    name = str(gate.get("name", "unknown"))
    remediation = REMEDIATION_BY_GATE.get(
        name,
        {
            "severity": "medium",
            "owner": "engineering",
            "required_fix": "Investigate the failing gate and add a documented remediation before approval.",
        },
    )
    return {
        "gate": name,
        "severity": remediation["severity"],
        "owner": remediation["owner"],
        "observed": gate.get("observed"),
        "threshold": gate.get("threshold"),
        "impact": gate.get("impact"),
        "required_fix": remediation["required_fix"],
    }


def _next_milestone(blocking_reasons: list[dict[str, Any]]) -> dict[str, str]:
    if not blocking_reasons:
        return {
            "name": "production rollout",
            "objective": "Run staged rollout with monitoring, rollback, and post-deployment calibration checks.",
        }
    if any(reason["gate"] == "dataset_audit_blockers" for reason in blocking_reasons):
        return {
            "name": "data quality unblock",
            "objective": "Resolve blocker-level dataset issues before treating model metrics as production evidence.",
        }
    if any(reason["gate"] in {"macro_f1", "balanced_accuracy", "accuracy_lift"} for reason in blocking_reasons):
        return {
            "name": "policy quality improvement",
            "objective": "Train and evaluate a stronger challenger model with class-aware objectives and richer features.",
        }
    return {
        "name": "targeted gate remediation",
        "objective": "Fix the remaining failing gates and regenerate the production readiness report.",
    }
