from pathlib import Path

from scripts.llm_training_delivery import (
    adapter_artifact_report,
    overall_status,
    read_baseline_comparison,
    validate_metric_gate,
)


def test_metric_gate_passes_when_thresholds_are_met() -> None:
    report = validate_metric_gate(
        {"accuracy": 0.9, "macro_f1": 0.8, "schema_validity_rate": 1.0, "unmatched_rate": 0.0},
        {
            "validation": {
                "min_accuracy": 0.5,
                "min_macro_f1": 0.5,
                "min_schema_validity_rate": 0.95,
                "max_unmatched_rate": 0.2,
            }
        },
    )
    assert report["status"] == "PASS"


def test_overall_status_prioritizes_failures() -> None:
    assert overall_status([{"status": "PASS"}, {"status": "FAIL"}]) == "FAIL"
    assert overall_status([{"status": "PASS"}, {"status": "WARN"}]) == "WARN"
    assert overall_status([{"status": "PASS"}]) == "PASS"


def test_adapter_artifact_report_detects_missing_adapter(tmp_path: Path) -> None:
    report = adapter_artifact_report(tmp_path)
    assert report["status"] == "FAIL"
    assert report["files"]["adapter_weights"]["exists"] is False


def test_read_baseline_comparison_selects_best_method(tmp_path: Path) -> None:
    path = tmp_path / "baseline.csv"
    path.write_text(
        "method,model,accuracy,macro_f1,schema_validity_rate\n"
        "zero_shot,m,0.5,0.4,1.0\n"
        "hybrid_parser_qlora,m,0.9,0.8,1.0\n",
        encoding="utf-8",
    )
    report = read_baseline_comparison(path)
    assert report["status"] == "PASS"
    assert report["best_method"]["method"] == "hybrid_parser_qlora"
