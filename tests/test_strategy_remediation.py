import json

from poker_agent.service import strategy_remediation_json
from poker_agent.strategy_remediation import build_strategy_remediation


def test_strategy_remediation_blocks_missing_timing_and_proxy_simulation(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "delivery_readiness.json").write_text(
        json.dumps({"service_delivery_status": "READY", "overall_status": "READY_FOR_TECHNICAL_HANDOFF"}),
        encoding="utf-8",
    )
    (reports / "policy_acceptance.json").write_text(
        json.dumps(
            {
                "overall_status": "FAIL",
                "human_action_alignment_status": "PASS",
                "simulation": {"status": "PASS", "simulation_type": "synthetic_policy_proxy"},
                "human_likeness": {
                    "status": "FAIL",
                    "js_divergence": 0.41,
                    "timing_and_bet_size_status": "NOT_AVAILABLE",
                    "timing_bet_size": {
                        "timing": {"status": "NOT_AVAILABLE"},
                        "bet_size": {"status": "FAIL", "js_divergence": 0.3},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "pdf_scope_alignment.json").write_text(
        json.dumps({"phase_statuses": {"phase_3_evaluation": "FAIL"}}),
        encoding="utf-8",
    )
    (reports / "production_gate.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")

    payload = build_strategy_remediation(tmp_path)
    blocker_ids = {item["id"] for item in payload["blocking_items"]}

    assert payload["strategy_policy_status"] == "NOT_APPROVED"
    assert payload["release_mode"] == "technical_handoff_only"
    assert "validated_self_play_environment" in blocker_ids
    assert "human_likeness_timing_proxy" in blocker_ids
    assert "human_likeness_bet_size_proxy" in blocker_ids
    assert "production_model_gate" in blocker_ids


def test_strategy_remediation_endpoint_exposes_workstreams() -> None:
    payload = strategy_remediation_json()

    assert payload["strategy_policy_status"] in {"APPROVED", "NOT_APPROVED", "UNKNOWN"}
    if payload["strategy_policy_status"] == "NOT_APPROVED":
        assert payload["blocking_items"]
        assert payload["engineering_workstreams"]


def test_production_scale_self_play_report_clears_scale_blocker(tmp_path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "delivery_readiness.json").write_text(
        json.dumps({"service_delivery_status": "READY", "overall_status": "READY_FOR_TECHNICAL_HANDOFF"}),
        encoding="utf-8",
    )
    (reports / "policy_acceptance.json").write_text(
        json.dumps(
            {
                "overall_status": "PASS",
                "human_action_alignment_status": "PASS",
                "simulation": {
                    "status": "PASS",
                    "simulation_type": "validated_multi_agent_holdem_self_play",
                    "runs": [{"hands": 100}],
                },
                "human_likeness": {
                    "status": "PASS",
                    "timing_and_bet_size_status": "PASS",
                    "timing_bet_size": {
                        "timing": {"status": "PASS"},
                        "bet_size": {"status": "PASS"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (reports / "production_self_play.json").write_text(
        json.dumps({"production_scale_status": "PASS", "paired_hands": 5000}),
        encoding="utf-8",
    )
    (reports / "pdf_scope_alignment.json").write_text(
        json.dumps({"phase_statuses": {"phase_3_evaluation": "PASS", "phase_4_deployment": "FAIL"}}),
        encoding="utf-8",
    )
    (reports / "production_gate.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")

    payload = build_strategy_remediation(tmp_path)
    blocker_ids = {item["id"] for item in payload["blocking_items"]}

    assert "production_scale_self_play" not in blocker_ids
    assert blocker_ids == {"production_model_gate"}
