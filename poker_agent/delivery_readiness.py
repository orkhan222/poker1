from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poker_agent.strategy_readiness import load_strategy_readiness


DELIVERY_READINESS_VERSION = "2026-06-19"


SERVICE_REQUIRED_CHECKS = {
    "required_files",
    "compile_sources",
    "model_loads",
    "inference_contract",
    "health_contract",
    "reports_contract",
    "repo_hygiene_contract",
    "hydra_provenance_contract",
    "zip_contract",
}


def summarize_delivery_readiness(project_root: Path) -> dict[str, Any]:
    reports_dir = project_root / "reports"
    delivery_verification = _read_json(reports_dir / "delivery_verification.json")
    repo_hygiene = _read_json(reports_dir / "repo_hygiene.json")
    scope_alignment = _read_json(reports_dir / "pdf_scope_alignment.json")
    strategy_readiness = load_strategy_readiness(reports_dir / "production_gate.json")

    verification_status = delivery_verification.get("status", "MISSING")
    hygiene_status = repo_hygiene.get("status", "MISSING")
    check_map = {
        str(check.get("name")): bool(check.get("passed"))
        for check in delivery_verification.get("checks", [])
        if check.get("name")
    }

    missing_checks = sorted(SERVICE_REQUIRED_CHECKS - set(check_map))
    failed_checks = sorted(name for name, passed in check_map.items() if name in SERVICE_REQUIRED_CHECKS and not passed)
    service_ready = (
        verification_status == "PASS"
        and hygiene_status == "PASS"
        and not missing_checks
        and not failed_checks
    )

    strategy_status = strategy_readiness.get("strategy_policy_status", "UNKNOWN")
    strategy_approved = strategy_status == "APPROVED"

    return {
        "version": DELIVERY_READINESS_VERSION,
        "overall_status": _overall_status(service_ready, strategy_approved),
        "service_delivery_status": "READY" if service_ready else "NOT_READY",
        "strategy_policy_status": strategy_status,
        "deployment_mode": (
            "production_policy"
            if service_ready and strategy_approved
            else "technical_handoff_only"
            if service_ready
            else "not_ready"
        ),
        "client_message": _client_message(service_ready, strategy_approved),
        "approval_boundary": {
            "service_delivery": "Confirms the API, model artifact loading, reports, hygiene, reproducibility checks, and ZIP package.",
            "strategy_policy": "Confirms the poker decision policy is strong enough for production strategy deployment.",
        },
        "service_evidence": {
            "delivery_verification": verification_status,
            "repo_hygiene": hygiene_status,
            "scope_alignment": scope_alignment.get("overall_status", "MISSING"),
            "required_checks": sorted(SERVICE_REQUIRED_CHECKS),
            "missing_checks": missing_checks,
            "failed_checks": failed_checks,
            "zip_contract": "PASS" if check_map.get("zip_contract") else "FAIL",
            "inference_contract": "PASS" if check_map.get("inference_contract") else "FAIL",
            "health_contract": "PASS" if check_map.get("health_contract") else "FAIL",
        },
        "strategy_evidence": {
            "production_gate_status": strategy_readiness.get("production_gate_status"),
            "scope_phase_statuses": scope_alignment.get("phase_statuses", {}),
            "metric_snapshot": strategy_readiness.get("metric_snapshot", {}),
            "blocking_reasons": strategy_readiness.get("blocking_reasons", []),
            "recommended_next_milestone": strategy_readiness.get("recommended_next_milestone", {}),
        },
    }


def write_delivery_readiness(project_root: Path, out_path: Path) -> dict[str, Any]:
    payload = summarize_delivery_readiness(project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _overall_status(service_ready: bool, strategy_approved: bool) -> str:
    if service_ready and strategy_approved:
        return "READY_FOR_PRODUCTION_POLICY"
    if service_ready:
        return "READY_FOR_TECHNICAL_HANDOFF"
    return "NOT_READY_FOR_HANDOFF"


def _client_message(service_ready: bool, strategy_approved: bool) -> str:
    if service_ready and strategy_approved:
        return "The service and strategy policy are approved for production deployment."
    if service_ready:
        return (
            "The service is ready for technical handoff, but the strategy model is not approved "
            "for production policy deployment."
        )
    return "The service is not ready for technical handoff until delivery checks pass."
