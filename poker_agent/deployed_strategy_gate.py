from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEPLOYED_STRATEGY_GATE_VERSION = "2026-06-20"
MIN_PRODUCTION_PAIRED_HANDS = 5000


def build_deployed_strategy_gate(project_root: Path) -> dict[str, Any]:
    reports_dir = project_root / "reports"
    policy_acceptance = _read_json(reports_dir / "policy_acceptance.json")
    production_self_play = _read_json(reports_dir / "production_self_play.json")
    raw_production_gate = _read_json(reports_dir / "production_gate.json")
    delivery_readiness = _read_json(reports_dir / "delivery_readiness.json")
    repo_hygiene = _read_json(reports_dir / "repo_hygiene.json")

    gates = [
        _gate(
            "policy_acceptance",
            policy_acceptance.get("overall_status") == "PASS",
            policy_acceptance.get("overall_status", "MISSING"),
            "PASS",
            "Deployment-gated policy must pass the acceptance suite.",
        ),
        _gate(
            "human_action_alignment",
            policy_acceptance.get("human_action_alignment_status") == "PASS",
            policy_acceptance.get("human_action_alignment_status", "MISSING"),
            "PASS",
            "Policy must align with held-out human action labels.",
        ),
        _gate(
            "human_likeness",
            (policy_acceptance.get("human_likeness") or {}).get("status") == "PASS",
            (policy_acceptance.get("human_likeness") or {}).get("status", "MISSING"),
            "PASS",
            "Action distribution, timing, and bet-size behavior must clear human-likeness gates.",
        ),
        _gate(
            "timing_and_bet_size",
            (policy_acceptance.get("human_likeness") or {}).get("timing_and_bet_size_status") == "PASS",
            (policy_acceptance.get("human_likeness") or {}).get("timing_and_bet_size_status", "MISSING"),
            "PASS",
            "Timing and bet-size proxies must be explicitly measured and pass.",
        ),
        _gate(
            "acceptance_self_play",
            (policy_acceptance.get("simulation") or {}).get("status") == "PASS"
            and (policy_acceptance.get("simulation") or {}).get("simulation_type")
            == "validated_multi_agent_holdem_self_play",
            {
                "status": (policy_acceptance.get("simulation") or {}).get("status", "MISSING"),
                "simulation_type": (policy_acceptance.get("simulation") or {}).get("simulation_type", "MISSING"),
            },
            {
                "status": "PASS",
                "simulation_type": "validated_multi_agent_holdem_self_play",
            },
            "Acceptance self-play must use the reviewed Hold'em simulator rather than a synthetic proxy.",
        ),
        _gate(
            "production_scale_self_play",
            _production_self_play_passes(production_self_play),
            {
                "status": production_self_play.get("status", "MISSING"),
                "production_scale_status": production_self_play.get("production_scale_status", "MISSING"),
                "paired_hands": _int(production_self_play.get("paired_hands")),
            },
            {
                "status": "PASS",
                "production_scale_status": "PASS",
                "paired_hands": f">={MIN_PRODUCTION_PAIRED_HANDS}",
            },
            "Production review requires a passing validated self-play run at production scale.",
        ),
        _gate(
            "service_delivery",
            delivery_readiness.get("service_delivery_status") == "READY",
            delivery_readiness.get("service_delivery_status", "MISSING"),
            "READY",
            "The API, packaging, reports, and reproducibility contract must be ready for handoff.",
        ),
        _gate(
            "repository_hygiene",
            repo_hygiene.get("status") == "PASS",
            repo_hygiene.get("status", "MISSING"),
            "PASS",
            "The delivery repository must pass hygiene checks before strategy approval is exposed.",
        ),
    ]

    blocking_items = [_blocker_from_gate(gate) for gate in gates if not gate["passed"]]
    raw_status = str(raw_production_gate.get("status", "MISSING")).upper()
    component_risks = _raw_model_component_risks(raw_production_gate)
    deployed_status = "PASS" if not blocking_items else "FAIL"

    return {
        "version": DEPLOYED_STRATEGY_GATE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": deployed_status,
        "strategy_policy_status": "APPROVED" if deployed_status == "PASS" else "NOT_APPROVED",
        "deployment_mode": "production_policy" if deployed_status == "PASS" else "technical_handoff_only",
        "production_claim_allowed": deployed_status == "PASS",
        "approval_boundary": {
            "deployed_strategy_stack": (
                "Approves the deployed policy stack: DeploymentGatedPolicy, action planner, "
                "human-alignment gate, human-likeness gate, and validated production-scale self-play."
            ),
            "raw_supervised_artifact": (
                "The standalone supervised artifact remains separately reported. It is not upgraded to PASS "
                "unless reports/production_gate.json passes on its own thresholds."
            ),
        },
        "decision": (
            "Approved for deployed strategy-stack rollout with monitoring."
            if deployed_status == "PASS"
            else "Blocked from deployed strategy-stack rollout until the failing deployed-stack gates pass."
        ),
        "raw_supervised_model_gate_status": raw_status,
        "raw_supervised_model_status": (
            "STANDALONE_APPROVED" if raw_status == "PASS" else "NOT_STANDALONE_APPROVED"
        ),
        "gates": gates,
        "blocking_items": blocking_items,
        "component_risks": component_risks,
        "metric_snapshot": _metric_snapshot(policy_acceptance, production_self_play, raw_production_gate),
        "source_reports": {
            "policy_acceptance": "reports/policy_acceptance.json",
            "production_self_play": "reports/production_self_play.json",
            "raw_production_gate": "reports/production_gate.json",
            "delivery_readiness": "reports/delivery_readiness.json",
            "repo_hygiene": "reports/repo_hygiene.json",
        },
        "recommended_next_milestone": _next_milestone(blocking_items, component_risks),
    }


def write_deployed_strategy_gate(
    project_root: Path,
    out_path: Path,
    markdown_out: Path | None = None,
) -> dict[str, Any]:
    payload = build_deployed_strategy_gate(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_deployed_strategy_gate_markdown(payload), encoding="utf-8")
    return payload


def render_deployed_strategy_gate_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Deployed Strategy Gate",
        "",
        "## Executive Status",
        "",
        f"- Status: `{payload['status']}`",
        f"- Strategy policy status: `{payload['strategy_policy_status']}`",
        f"- Deployment mode: `{payload['deployment_mode']}`",
        f"- Production claim allowed: `{payload['production_claim_allowed']}`",
        f"- Raw supervised model status: `{payload['raw_supervised_model_status']}`",
        "",
        payload["decision"],
        "",
        "## Deployed Stack Gates",
        "",
        "| Gate | Passed | Observed | Threshold |",
        "| --- | --- | --- | --- |",
    ]
    for gate in payload["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['passed']} | `{_markdown_value(gate['observed'])}` | "
            f"`{_markdown_value(gate['threshold'])}` |"
        )
    lines.extend(["", "## Component Risks", "", "| Component | Status | Evidence |"])
    lines.append("| --- | --- | --- |")
    if payload["component_risks"]:
        for risk in payload["component_risks"]:
            lines.append(f"| {risk['component']} | {risk['status']} | {risk['evidence']} |")
    else:
        lines.append("| none | none | No component risks are currently reported. |")
    lines.append("")
    return "\n".join(lines)


def _gate(name: str, passed: bool, observed: Any, threshold: Any, impact: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
        "impact": impact,
    }


def _blocker_from_gate(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": gate["name"],
        "severity": "critical",
        "evidence": f"observed={_compact_value(gate['observed'])}, threshold={_compact_value(gate['threshold'])}",
        "required_fix": gate["impact"],
    }


def _raw_model_component_risks(raw_production_gate: dict[str, Any]) -> list[dict[str, Any]]:
    raw_status = str(raw_production_gate.get("status", "MISSING")).upper()
    if raw_status == "PASS":
        return []
    readiness = raw_production_gate.get("strategy_readiness") or {}
    reasons = readiness.get("blocking_reasons") or []
    evidence = ", ".join(
        f"{reason.get('gate')}={_compact_value(reason.get('observed'))}"
        for reason in reasons[:6]
        if reason.get("gate")
    )
    if not evidence:
        evidence = f"production_gate={raw_status}"
    return [
        {
            "component": "raw_supervised_model_artifact",
            "status": "NOT_STANDALONE_APPROVED",
            "evidence": evidence,
            "impact": (
                "The raw supervised artifact should not be marketed as a standalone production strategy model. "
                "It is wrapped by the deployed strategy stack and remains tracked as a model-quality risk."
            ),
            "required_fix": (
                "Train and gate a challenger artifact that clears macro-F1, balanced-accuracy, lift, "
                "facing-bet, observed-card, and dataset-audit thresholds."
            ),
        }
    ]


def _metric_snapshot(
    policy_acceptance: dict[str, Any],
    production_self_play: dict[str, Any],
    raw_production_gate: dict[str, Any],
) -> dict[str, Any]:
    alignment = policy_acceptance.get("human_action_alignment") or {}
    likeness = policy_acceptance.get("human_likeness") or {}
    raw_valid = raw_production_gate.get("valid_metrics") or {}
    return {
        "human_action_accuracy": alignment.get("accuracy"),
        "human_action_macro_f1": alignment.get("macro_f1"),
        "human_action_lift_vs_majority": alignment.get("lift_vs_majority"),
        "human_likeness_js_divergence": likeness.get("js_divergence"),
        "production_self_play_paired_hands": production_self_play.get("paired_hands"),
        "production_self_play_mean_win_rate": production_self_play.get("mean_policy_win_rate"),
        "production_self_play_p_win_rate_ge_50_5": production_self_play.get(
            "probability_win_rate_at_least_50_5"
        ),
        "production_self_play_p_win_rate_ge_52": production_self_play.get(
            "probability_win_rate_at_least_52"
        ),
        "raw_model_macro_f1": raw_valid.get("macro_f1"),
        "raw_model_balanced_accuracy": raw_valid.get("balanced_accuracy"),
        "raw_model_accuracy_lift": raw_valid.get("lift_vs_majority"),
    }


def _production_self_play_passes(report: dict[str, Any]) -> bool:
    return (
        report.get("status") == "PASS"
        and report.get("production_scale_status") == "PASS"
        and _int(report.get("paired_hands")) >= MIN_PRODUCTION_PAIRED_HANDS
    )


def _next_milestone(blocking_items: list[dict[str, Any]], component_risks: list[dict[str, Any]]) -> dict[str, str]:
    if blocking_items:
        return {
            "name": "deployed stack gate remediation",
            "objective": "Clear the failing deployed-stack gates before exposing production strategy approval.",
        }
    if component_risks:
        return {
            "name": "standalone challenger artifact",
            "objective": "Train a new supervised artifact that clears the raw production model gate without relying on stack-level routing.",
        }
    return {
        "name": "staged production rollout",
        "objective": "Deploy with monitoring, rollback, and periodic re-gating of the raw artifact and deployed stack.",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _markdown_value(value: Any) -> str:
    return _compact_value(value).replace("|", "\\|")
