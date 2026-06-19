from scripts.evaluate_llm_context_ablation import (
    best_profile_row,
    classification_metrics,
    comparison_row,
    parse_profiles,
    profile_few_shot_count,
    profile_mode,
)


def test_parse_profiles_rejects_unknown_profile() -> None:
    try:
        parse_profiles("minimal,unknown")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("Expected unsupported prompt profile to fail")


def test_profile_mode_maps_candidate_ranker_only() -> None:
    assert profile_mode("candidate_ranker") == "candidate_ranker"
    assert profile_mode("rules_zero_shot") == "freeform"


def test_few_shot_count_only_applies_to_contextual_profiles() -> None:
    assert profile_few_shot_count("minimal", 5) == 0
    assert profile_few_shot_count("rules_zero_shot", 5) == 0
    assert profile_few_shot_count("rules_few_shot", 5) == 5
    assert profile_few_shot_count("candidate_ranker", 5) == 5


def test_classification_metrics_reports_majority_lift() -> None:
    metrics = classification_metrics(
        ["fold", "call", "call"],
        ["fold", "fold", "call"],
        [
            {"fold": 0.8, "call": 0.2},
            {"fold": 0.6, "call": 0.4},
            {"fold": 0.3, "call": 0.7},
        ],
    )
    assert metrics["accuracy"] == 2 / 3
    assert metrics["majority_baseline_accuracy"] == 2 / 3
    assert metrics["examples"] == 3.0


def test_comparison_row_flattens_nested_summary() -> None:
    row = comparison_row(
        {
            "profile": "rules_few_shot",
            "mode": "freeform",
            "prompt_contract": {"rules_count": 10, "guidelines_count": 7, "few_shot_count": 5},
            "metrics": {
                "examples": 12.0,
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "weighted_f1": 0.45,
                "cross_entropy": 1.2,
            },
            "latency_ms": {"mean": 3.0},
            "prompt_chars": {"mean": 1500.0},
            "invalid_output_rate": 0.1,
        },
        provider="heuristic_text",
        model_id="local_scoring_policy",
    )
    assert row["profile"] == "rules_few_shot"
    assert row["few_shot_count"] == 5
    assert row["mean_prompt_chars"] == 1500.0


def test_best_profile_prefers_candidate_ranker_on_metric_tie() -> None:
    rows = [
        {"profile": "minimal", "macro_f1": 0.2, "accuracy": 0.3, "invalid_output_rate": 0.0},
        {"profile": "candidate_ranker", "macro_f1": 0.2, "accuracy": 0.3, "invalid_output_rate": 0.0},
    ]
    assert best_profile_row(rows)["profile"] == "candidate_ranker"
