from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.model import load_policy
from poker_agent.strategy_readiness import summarize_strategy_readiness


DEFAULT_THRESHOLDS = {
    "min_accuracy_lift": 0.0,
    "min_macro_f1": 0.50,
    "min_balanced_accuracy": 0.50,
    "max_ece_10": 0.10,
    "min_observed_hole_macro_f1": 0.50,
    "min_facing_bet_macro_f1": 0.45,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model artifact against production ML gates")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--audit-report", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/production_gate.json"))
    parser.add_argument("--min-accuracy-lift", type=float, default=DEFAULT_THRESHOLDS["min_accuracy_lift"])
    parser.add_argument("--min-macro-f1", type=float, default=DEFAULT_THRESHOLDS["min_macro_f1"])
    parser.add_argument("--min-balanced-accuracy", type=float, default=DEFAULT_THRESHOLDS["min_balanced_accuracy"])
    parser.add_argument("--max-ece-10", type=float, default=DEFAULT_THRESHOLDS["max_ece_10"])
    parser.add_argument("--min-observed-hole-macro-f1", type=float, default=DEFAULT_THRESHOLDS["min_observed_hole_macro_f1"])
    parser.add_argument("--min-facing-bet-macro-f1", type=float, default=DEFAULT_THRESHOLDS["min_facing_bet_macro_f1"])
    return parser.parse_args()


def gate_result(name: str, passed: bool, observed: Any, threshold: Any, impact: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "impact": impact,
    }


def audit_findings(audit_report: Path | None) -> list[dict[str, Any]]:
    if audit_report is None or not audit_report.exists():
        return []
    payload = json.loads(audit_report.read_text(encoding="utf-8"))
    return list(payload.get("findings", []))


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = load_policy(args.model)
    metadata = getattr(model, "metadata", {}) or {}
    valid = metadata.get("valid_metrics", {})
    slices = metadata.get("valid_slice_metrics", {})
    split = metadata.get("split", {})
    findings = audit_findings(args.audit_report)

    gates: list[dict[str, Any]] = []
    gates.append(
        gate_result(
            "validation_split",
            split.get("split_type") in {"stratified_hand_group_holdout", "session_holdout", "source_holdout"},
            split.get("split_type"),
            "group/session/source holdout",
            "Prevents optimistic action-level leakage.",
        )
    )
    gates.append(
        gate_result(
            "accuracy_lift",
            float(valid.get("lift_vs_majority", -999.0)) >= args.min_accuracy_lift,
            valid.get("lift_vs_majority"),
            args.min_accuracy_lift,
            "Model must beat the majority-class baseline on the same split.",
        )
    )
    gates.append(
        gate_result(
            "macro_f1",
            float(valid.get("macro_f1", 0.0)) >= args.min_macro_f1,
            valid.get("macro_f1"),
            args.min_macro_f1,
            "Minority poker actions must be learned, not hidden by fold dominance.",
        )
    )
    gates.append(
        gate_result(
            "balanced_accuracy",
            float(valid.get("balanced_accuracy", 0.0)) >= args.min_balanced_accuracy,
            valid.get("balanced_accuracy"),
            args.min_balanced_accuracy,
            "Recall must be acceptable across action classes.",
        )
    )
    gates.append(
        gate_result(
            "calibration",
            float(valid.get("ece_10", 999.0)) <= args.max_ece_10,
            valid.get("ece_10"),
            args.max_ece_10,
            "Prediction confidence must be reviewable for downstream consumers.",
        )
    )

    observed_hole = slices.get("observed_hole_cards", {})
    if observed_hole:
        gates.append(
            gate_result(
                "observed_hole_cards_macro_f1",
                float(observed_hole.get("macro_f1", 0.0)) >= args.min_observed_hole_macro_f1,
                observed_hole.get("macro_f1"),
                args.min_observed_hole_macro_f1,
                "The model must perform on the slice where card signal exists.",
            )
        )

    facing_bet = slices.get("facing_bet", {})
    if facing_bet:
        gates.append(
            gate_result(
                "facing_bet_macro_f1",
                float(facing_bet.get("macro_f1", 0.0)) >= args.min_facing_bet_macro_f1,
                facing_bet.get("macro_f1"),
                args.min_facing_bet_macro_f1,
                "Call/fold/raise decisions under pressure are the core business case.",
            )
        )

    blocker_findings = [finding for finding in findings if finding.get("severity") == "blocker"]
    gates.append(
        gate_result(
            "dataset_audit_blockers",
            not blocker_findings,
            len(blocker_findings),
            0,
            "No production model should pass while dataset audit has blocker findings.",
        )
    )

    passed = all(gate["passed"] for gate in gates)
    report = {
        "status": "PASS" if passed else "FAIL",
        "model": str(args.model),
        "policy": metadata.get("policy", getattr(model, "model_kind", "unknown")),
        "split": split,
        "valid_metrics": valid,
        "gates": gates,
        "audit_findings": findings,
        "decision": (
            "Approved for production decision-policy deployment."
            if passed
            else "Not approved for production decision-policy deployment."
        ),
    }
    report["strategy_readiness"] = summarize_strategy_readiness(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "strategy_policy_status": report["strategy_readiness"]["strategy_policy_status"],
                "deployment_mode": report["strategy_readiness"]["deployment_mode"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
