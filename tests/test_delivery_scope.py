from poker_agent.delivery_scope import (
    ScopeGate,
    model_row_from_metrics,
    phase_status,
    status_from_metric,
)


def test_status_from_metric_uses_threshold_direction() -> None:
    assert status_from_metric(0.75, 0.50) == "PASS"
    assert status_from_metric(0.25, 0.50) == "FAIL"
    assert status_from_metric(0.05, 0.10, higher_is_better=False) == "PASS"
    assert status_from_metric(0.20, 0.10, higher_is_better=False) == "FAIL"


def test_phase_status_marks_failure_when_any_gate_fails() -> None:
    gates = [
        ScopeGate("policy", "PASS", "ok", "none", "continue"),
        ScopeGate("simulation", "FAIL", "missing", "approval risk", "implement simulation"),
    ]
    assert phase_status(gates) == "FAIL"


def test_model_row_from_metrics_normalizes_missing_values() -> None:
    row = model_row_from_metrics(
        system="Policy",
        family="hist_gradient_boosting",
        metrics={"accuracy": 0.67891, "macro_f1": 0.41234},
        latency_ms=None,
        status="FAIL",
        notes="below gate",
    ).to_dict()
    assert row["accuracy"] == 0.6789
    assert row["macro_f1"] == 0.4123
    assert row["weighted_f1"] == 0.0
    assert row["latency_ms"] is None
