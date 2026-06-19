import json

from poker_agent.delivery_readiness import summarize_delivery_readiness
from poker_agent.service import delivery_readiness_json


def test_delivery_readiness_separates_handoff_from_strategy_approval(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    checks = [
        {"name": name, "passed": True}
        for name in {
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
    ]
    (reports / "delivery_verification.json").write_text(
        json.dumps({"status": "PASS", "checks": checks}),
        encoding="utf-8",
    )
    (reports / "repo_hygiene.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (reports / "production_gate.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "valid_metrics": {"macro_f1": 0.41},
                "gates": [
                    {
                        "name": "macro_f1",
                        "passed": False,
                        "observed": 0.41,
                        "threshold": 0.5,
                        "impact": "Minority actions are not learned strongly enough.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = summarize_delivery_readiness(tmp_path)

    assert payload["overall_status"] == "READY_FOR_TECHNICAL_HANDOFF"
    assert payload["service_delivery_status"] == "READY"
    assert payload["strategy_policy_status"] == "NOT_APPROVED"
    assert payload["deployment_mode"] == "technical_handoff_only"
    assert payload["strategy_evidence"]["blocking_reasons"]


def test_delivery_readiness_endpoint_exposes_client_message() -> None:
    payload = delivery_readiness_json()

    assert payload["overall_status"] in {
        "READY_FOR_TECHNICAL_HANDOFF",
        "READY_FOR_PRODUCTION_POLICY",
        "NOT_READY_FOR_HANDOFF",
    }
    assert payload["client_message"]
    assert "service_delivery_status" in payload
    assert "strategy_policy_status" in payload
