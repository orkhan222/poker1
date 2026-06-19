from poker_agent.service import scope_alignment_json


def test_scope_alignment_endpoint_exposes_client_scope_report() -> None:
    payload = scope_alignment_json()

    assert payload["overall_status"] in {"PASS", "FAIL", "PARTIAL", "MISSING"}
    if payload["overall_status"] != "MISSING":
        assert "phase_statuses" in payload
        assert "model_comparison" in payload
        assert payload["recommendation"]["event_normalization_agent"]
