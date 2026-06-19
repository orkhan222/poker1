from poker_agent.service import strategy_readiness_json
from poker_agent.strategy_readiness import summarize_strategy_readiness


def test_failing_production_gate_becomes_not_approved_readiness() -> None:
    report = {
        "status": "FAIL",
        "valid_metrics": {
            "accuracy": 0.67,
            "majority_baseline_accuracy": 0.70,
            "lift_vs_majority": -0.03,
            "macro_f1": 0.41,
            "balanced_accuracy": 0.44,
        },
        "gates": [
            {
                "name": "accuracy_lift",
                "passed": False,
                "observed": -0.03,
                "threshold": 0.0,
                "impact": "Model must beat the majority-class baseline on the same split.",
            },
            {
                "name": "calibration",
                "passed": True,
                "observed": 0.07,
                "threshold": 0.1,
                "impact": "Prediction confidence must be reviewable.",
            },
        ],
    }

    readiness = summarize_strategy_readiness(report)

    assert readiness["strategy_policy_status"] == "NOT_APPROVED"
    assert readiness["deployment_mode"] == "technical_handoff_only"
    assert readiness["blocking_reasons"][0]["gate"] == "accuracy_lift"
    assert readiness["blocking_reasons"][0]["required_fix"]


def test_passing_production_gate_becomes_approved_readiness() -> None:
    report = {
        "status": "PASS",
        "valid_metrics": {"macro_f1": 0.62, "balanced_accuracy": 0.58},
        "gates": [{"name": "macro_f1", "passed": True, "observed": 0.62, "threshold": 0.5}],
    }

    readiness = summarize_strategy_readiness(report)

    assert readiness["strategy_policy_status"] == "APPROVED"
    assert readiness["deployment_mode"] == "production_policy"
    assert readiness["blocking_reasons"] == []


def test_strategy_readiness_endpoint_exposes_blocking_reasons() -> None:
    payload = strategy_readiness_json()

    assert payload["strategy_policy_status"] in {"APPROVED", "NOT_APPROVED", "UNKNOWN"}
    if payload["strategy_policy_status"] == "NOT_APPROVED":
        assert payload["blocking_reasons"]
        assert payload["recommended_next_milestone"]["objective"]
