from poker_agent.api_contract import api_contract, prediction_response_contract
from poker_agent.service import contract_json


def test_prediction_response_contract_documents_required_fields() -> None:
    fields = prediction_response_contract()["response_fields"]
    for field in (
        "action",
        "probabilities",
        "confidence",
        "bet_size",
        "wait_time_ms",
        "sizing_method",
        "timing_method",
        "model_status",
    ):
        assert field in fields
        assert fields[field]["description"]


def test_contract_endpoint_exposes_delivery_status_terms() -> None:
    payload = contract_json()
    statuses = payload["delivery_status"]["delivery_status_fields"]
    assert "delivery_verification=PASS" in statuses
    assert "production_gate=FAIL" in statuses


def test_api_contract_separates_delivery_from_strategy_approval() -> None:
    payload = api_contract()
    boundary = payload["approval_boundary"]
    assert "software_delivery" in boundary
    assert "strategy_model" in boundary
    assert payload["strategy_readiness"]["endpoint"] == "/strategy-readiness.json"
    assert "NOT_APPROVED" in payload["strategy_readiness"]["status_values"]
    assert payload["delivery_readiness"]["endpoint"] == "/delivery-readiness.json"
    assert "READY_FOR_TECHNICAL_HANDOFF" in payload["delivery_readiness"]["overall_status_values"]
    assert payload["scope_alignment"]["endpoint"] == "/scope-alignment.json"
    assert "Poker ML Project.docx" in payload["scope_alignment"]["source_documents"]
