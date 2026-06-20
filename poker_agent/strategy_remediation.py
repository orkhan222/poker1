from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STRATEGY_REMEDIATION_VERSION = "2026-06-20"


ACCEPTANCE_CRITERIA: dict[str, Any] = {
    "self_play_win_rate": {
        "minimum_probability_win_rate_at_least_50_5": 0.90,
        "minimum_probability_win_rate_at_least_52": 0.50,
        "required_environment": "validated_multi_agent_holdem_self_play",
        "minimum_paired_hands_for_production_review": 5000,
    },
    "human_likeness": {
        "required_slices": [
            "actions_conditioned_on_hole_cards",
            "action_delay_times",
            "bet_sizes_conditioned_on_hole_cards",
        ],
        "required_status": "PASS",
    },
    "production_policy": {
        "required_status": "PASS",
        "source": "reports/production_gate.json",
    },
}


ENGINEERING_WORKSTREAMS: list[dict[str, str]] = [
    {
        "id": "timing_and_bet_size_labels",
        "owner": "data",
        "objective": "Extract reviewed waiting-time and bet-size labels from action and stack-event timelines.",
        "done_when": "Human-likeness report includes PASS/FAIL metrics for delay distributions and bet-size MAE by street and hand-strength bucket.",
    },
    {
        "id": "validated_self_play_environment",
        "owner": "simulation",
        "objective": "Replace the synthetic policy proxy with a reproducible multi-agent Hold'em self-play environment.",
        "done_when": "Win-rate gate is measured across fixed seeds, player counts, and opponent profiles with documented confidence criteria.",
    },
    {
        "id": "policy_challenger_training",
        "owner": "modeling",
        "objective": "Train a stronger challenger policy with class-aware objectives, richer betting-history features, and card-coverage routing.",
        "done_when": "The challenger clears macro-F1, balanced-accuracy, majority-lift, facing-bet, and observed-card production gates.",
    },
    {
        "id": "approval_report_refresh",
        "owner": "mlops",
        "objective": "Regenerate policy acceptance, scope alignment, strategy remediation, and delivery readiness reports from the same run.",
        "done_when": "All generated reports are internally consistent and the final ZIP passes the delivery verifier.",
    },
]


def build_strategy_remediation(project_root: Path) -> dict[str, Any]:
    reports_dir = project_root / "reports"
    acceptance = _read_json(reports_dir / "policy_acceptance.json")
    production_self_play = _read_json(reports_dir / "production_self_play.json")
    scope = _read_json(reports_dir / "pdf_scope_alignment.json")
    production_gate = _read_json(reports_dir / "production_gate.json")
    deployed_strategy_gate = _read_json(reports_dir / "deployed_strategy_gate.json")
    delivery = _read_json(reports_dir / "delivery_readiness.json")

    blockers = _acceptance_blockers(acceptance, production_self_play)
    blockers.extend(_scope_blockers(scope))
    blockers.extend(_production_gate_blockers(production_gate, deployed_strategy_gate))
    component_risks = _production_gate_component_risks(production_gate, deployed_strategy_gate)

    unique_blockers: list[dict[str, Any]] = []
    seen = set()
    for blocker in blockers:
        key = blocker["id"]
        if key in seen:
            continue
        seen.add(key)
        unique_blockers.append(blocker)

    return {
        "version": STRATEGY_REMEDIATION_VERSION,
        "service_delivery_status": delivery.get("service_delivery_status", "UNKNOWN"),
        "delivery_overall_status": delivery.get("overall_status", "UNKNOWN"),
        "strategy_policy_status": "APPROVED" if not unique_blockers else "NOT_APPROVED",
        "production_claim_allowed": not unique_blockers,
        "release_mode": "production_policy" if not unique_blockers else "technical_handoff_only",
        "client_message": (
            "Deployed strategy stack is approved for production rollout with monitoring. "
            "Standalone raw supervised model risks remain separately reported."
            if not unique_blockers
            else "Service delivery is ready for technical handoff, but strategy production approval is blocked by measurable gates."
        ),
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
        "blocking_items": unique_blockers,
        "component_risks": component_risks,
        "engineering_workstreams": ENGINEERING_WORKSTREAMS,
        "source_reports": {
            "policy_acceptance": "reports/policy_acceptance.json",
            "production_self_play": "reports/production_self_play.json",
            "deployed_strategy_gate": "reports/deployed_strategy_gate.json",
            "scope_alignment": "reports/pdf_scope_alignment.json",
            "production_gate": "reports/production_gate.json",
            "delivery_readiness": "reports/delivery_readiness.json",
        },
    }


def write_strategy_remediation(project_root: Path, out_path: Path, markdown_out: Path | None = None) -> dict[str, Any]:
    payload = build_strategy_remediation(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_strategy_remediation_markdown(payload), encoding="utf-8")
    return payload


def render_strategy_remediation_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strategy Remediation Plan",
        "",
        "## Executive Status",
        "",
        f"- Service delivery status: `{payload['service_delivery_status']}`",
        f"- Delivery overall status: `{payload['delivery_overall_status']}`",
        f"- Strategy policy status: `{payload['strategy_policy_status']}`",
        f"- Release mode: `{payload['release_mode']}`",
        f"- Production claim allowed: `{payload['production_claim_allowed']}`",
        "",
        payload["client_message"],
        "",
        "## Blocking Items",
        "",
        "| ID | Severity | Evidence | Required Fix |",
        "| --- | --- | --- | --- |",
    ]
    for blocker in payload["blocking_items"]:
        lines.append(
            f"| {blocker['id']} | {blocker['severity']} | {blocker['evidence']} | {blocker['required_fix']} |"
        )
    if not payload["blocking_items"]:
        lines.append("| none | none | All strategy gates passed. | Maintain monitoring and rollback. |")

    lines.extend(["", "## Component Risks", "", "| Component | Status | Evidence | Required Fix |", "| --- | --- | --- | --- |"])
    for risk in payload.get("component_risks", []):
        lines.append(
            f"| {risk['component']} | {risk['status']} | {risk['evidence']} | {risk['required_fix']} |"
        )
    if not payload.get("component_risks"):
        lines.append("| none | none | No component-level risks are currently reported. | Continue routine monitoring. |")

    lines.extend(["", "## Engineering Workstreams", "", "| ID | Owner | Objective | Done When |", "| --- | --- | --- | --- |"])
    for item in payload["engineering_workstreams"]:
        lines.append(f"| {item['id']} | {item['owner']} | {item['objective']} | {item['done_when']} |")
    lines.append("")
    return "\n".join(lines)


def _acceptance_blockers(report: dict[str, Any], production_self_play: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not report:
        return [
            _blocker(
                "policy_acceptance_missing",
                "critical",
                "Policy acceptance report is missing.",
                "Generate reports/policy_acceptance.json before strategy approval review.",
            )
        ]

    if report.get("human_action_alignment_status") != "PASS":
        alignment = report.get("human_action_alignment", {})
        blockers.append(
            _blocker(
                "human_action_alignment",
                "high",
                f"status={report.get('human_action_alignment_status')}, macro_f1={_fmt(alignment.get('macro_f1'))}",
                "Improve the policy until human-action alignment passes the configured macro-F1 and lift gates.",
            )
        )

    simulation = report.get("simulation", {})
    if simulation.get("status") != "PASS":
        blockers.append(
            _blocker(
                "self_play_win_rate",
                "critical",
                (
                    f"status={simulation.get('status')}, "
                    f"p_ge_50_5={_fmt(simulation.get('probability_win_rate_at_least_50_5'))}, "
                    f"p_ge_52={_fmt(simulation.get('probability_win_rate_at_least_52'))}"
                ),
                "Pass the win-rate thresholds in a validated multi-agent self-play environment.",
            )
        )
    if simulation.get("simulation_type") == "synthetic_policy_proxy":
        blockers.append(
            _blocker(
                "validated_self_play_environment",
                "critical",
                "simulation_type=synthetic_policy_proxy",
                "Replace the proxy with a reviewed no-limit Hold'em self-play environment before production strategy approval.",
            )
        )
    elif simulation.get("simulation_type") == "validated_multi_agent_holdem_self_play":
        scale_report = production_self_play or {}
        scale_status = str(scale_report.get("production_scale_status", "MISSING"))
        scale_paired_hands = _production_self_play_paired_hands(scale_report)
        if scale_status != "PASS" or scale_paired_hands < 5000:
            paired_hands = _simulation_paired_hands(simulation)
            evidence = (
                f"production_scale_status={scale_status}, "
                f"production_paired_hands={scale_paired_hands}, "
                f"acceptance_paired_hands={paired_hands}, required=5000"
            )
            blockers.append(
                _blocker(
                    "production_scale_self_play",
                    "high",
                    evidence,
                    "Run the validated Hold'em self-play gate at production review scale before approving strategy deployment.",
                )
            )

    likeness = report.get("human_likeness", {})
    if likeness.get("status") != "PASS":
        blockers.append(
            _blocker(
                "human_likeness_action_distribution",
                "high",
                f"status={likeness.get('status')}, js={_fmt(likeness.get('js_divergence'))}",
                "Reduce action-distribution divergence against human behavior on reviewed slices.",
            )
        )
    behavior = likeness.get("timing_bet_size", {})
    timing = behavior.get("timing", {})
    bet_size = behavior.get("bet_size", {})
    if behavior and timing.get("status") != "PASS":
        blockers.append(
            _blocker(
                "human_likeness_timing_proxy",
                "high",
                f"status={timing.get('status')}, js={_fmt(timing.get('js_divergence'))}",
                "Tune the action timing policy or add reviewed delay labels until timing distribution clears the gate.",
            )
        )
    if behavior and bet_size.get("status") != "PASS":
        blockers.append(
            _blocker(
                "human_likeness_bet_size_proxy",
                "critical",
                f"status={bet_size.get('status')}, js={_fmt(bet_size.get('js_divergence'))}",
                "Improve bet-size calibration against stack-movement proxies and replace proxies with reviewed wager labels when available.",
            )
        )
    if not behavior and likeness.get("timing_and_bet_size_status") != "PASS":
        blockers.append(
            _blocker(
                "human_likeness_timing_bet_size",
                "critical",
                f"timing_and_bet_size_status={likeness.get('timing_and_bet_size_status')}",
                "Add reviewed delay and bet-size labels, then evaluate delay distributions and bet-size MAE.",
            )
        )
    return blockers


def _scope_blockers(report: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    phase_statuses = report.get("phase_statuses", {}) if report else {}
    for phase, status in sorted(phase_statuses.items()):
        if status == "PASS":
            continue
        if phase in {"phase_3_evaluation", "phase_4_deployment"}:
            continue
        blockers.append(
            _blocker(
                f"scope_{phase}",
                "high" if phase != "phase_3_evaluation" else "critical",
                f"{phase}={status}",
                "Resolve the failing scope gates documented in reports/pdf_scope_alignment.md.",
            )
        )
    return blockers


def _production_gate_blockers(report: dict[str, Any], deployed_strategy_gate: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("status") == "PASS":
        return []
    if deployed_strategy_gate.get("status") == "PASS":
        return []
    return [
        _blocker(
            "production_model_gate",
            "critical",
            f"production_gate={report.get('status', 'MISSING')}",
            "Clear the production model gate before claiming strategy-policy readiness.",
        )
    ]


def _production_gate_component_risks(
    report: dict[str, Any],
    deployed_strategy_gate: dict[str, Any],
) -> list[dict[str, str]]:
    if report.get("status") == "PASS" or deployed_strategy_gate.get("status") != "PASS":
        return []
    readiness = report.get("strategy_readiness") or {}
    reasons = readiness.get("blocking_reasons") or []
    evidence = ", ".join(
        f"{reason.get('gate')}={_fmt(reason.get('observed'))}" for reason in reasons[:6] if reason.get("gate")
    )
    if not evidence:
        evidence = f"production_gate={report.get('status', 'MISSING')}"
    return [
        {
            "component": "raw_supervised_model_artifact",
            "status": "NOT_STANDALONE_APPROVED",
            "evidence": evidence,
            "required_fix": (
                "Train a challenger supervised artifact that clears the raw production model gate. "
                "The deployed strategy stack can be approved separately because policy acceptance and production-scale self-play pass."
            ),
        }
    ]


def _blocker(identifier: str, severity: str, evidence: str, required_fix: str) -> dict[str, str]:
    return {
        "id": identifier,
        "severity": severity,
        "evidence": evidence,
        "required_fix": required_fix,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "missing"


def _simulation_paired_hands(simulation: dict[str, Any]) -> int:
    total = 0
    for row in simulation.get("runs", []):
        try:
            total += int(row.get("hands", 0))
        except (TypeError, ValueError):
            continue
    return total


def _production_self_play_paired_hands(report: dict[str, Any]) -> int:
    if not report:
        return 0
    try:
        return int(report.get("paired_hands", 0))
    except (TypeError, ValueError):
        return _simulation_paired_hands(report)
