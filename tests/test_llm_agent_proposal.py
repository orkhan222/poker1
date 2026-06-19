from pathlib import Path

from poker_agent.llm_agent_proposal import (
    adapter_artifact_available,
    build_recommendation,
    default_architecture_options,
)


def test_recommendation_selects_hybrid_architecture_from_decision_payload(tmp_path: Path) -> None:
    model_dir = tmp_path / "adapter"
    model_dir.mkdir()
    (model_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    recommendation = build_recommendation(
        config={"base_model": "Qwen/Qwen2.5-1.5B-Instruct"},
        decision_payload={
            "selected_architecture": "hybrid_parser_qlora",
            "comparison": [
                {
                    "architecture": "hybrid_parser_qlora",
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "schema_validity_rate": 1.0,
                }
            ],
        },
        model_dir=model_dir,
        approval_status="proposed_for_stakeholder_approval",
    )

    assert recommendation.selected_architecture == "hybrid_parser_qlora"
    assert recommendation.selected_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert recommendation.adapter_available is True
    assert recommendation.ready_for_stakeholder_review is True
    assert recommendation.architecture_metrics["macro_f1"] == 1.0


def test_adapter_artifact_requires_weights_and_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "adapter"
    model_dir.mkdir()
    (model_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    assert adapter_artifact_available(model_dir) is False
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert adapter_artifact_available(model_dir) is True


def test_architecture_options_include_non_recommended_autonomous_agent() -> None:
    options = {option.name: option for option in default_architecture_options()}
    assert options["hybrid_parser_qlora"].production_fit.startswith("Recommended")
    assert options["autonomous_decision_llm"].operational_risk == "High"
