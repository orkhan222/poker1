from poker_agent.policy_acceptance import (
    EVCalibratedPolicy,
    aggregate_simulations,
    ev_calibrated_probabilities,
    jensen_shannon_divergence,
    normalize_distribution,
    total_variation_distance,
)


def test_distribution_distances_are_zero_for_identical_distributions() -> None:
    left = normalize_distribution({"fold": 3, "call": 2})
    right = normalize_distribution({"fold": 3, "call": 2})
    assert jensen_shannon_divergence(left, right) == 0.0
    assert total_variation_distance(left, right) == 0.0


def test_distribution_distances_increase_for_different_distributions() -> None:
    left = normalize_distribution({"fold": 10})
    right = normalize_distribution({"raise": 10})
    assert jensen_shannon_divergence(left, right) > 0.5
    assert total_variation_distance(left, right) == 1.0


def test_aggregate_simulations_enforces_pdf_win_rate_thresholds() -> None:
    passing = aggregate_simulations(
        [
            {"policy_win_rate": 0.53, "mean_ev_delta_vs_baseline": 0.1},
            {"policy_win_rate": 0.52, "mean_ev_delta_vs_baseline": 0.0},
            {"policy_win_rate": 0.51, "mean_ev_delta_vs_baseline": -0.1},
            {"policy_win_rate": 0.53, "mean_ev_delta_vs_baseline": 0.2},
            {"policy_win_rate": 0.52, "mean_ev_delta_vs_baseline": 0.1},
        ],
        min_win_rate_high_confidence=0.505,
        min_win_rate_median=0.52,
    )
    assert passing["status"] == "PASS"

    failing = aggregate_simulations(
        [
            {"policy_win_rate": 0.50, "mean_ev_delta_vs_baseline": 0.0},
            {"policy_win_rate": 0.51, "mean_ev_delta_vs_baseline": 0.0},
        ],
        min_win_rate_high_confidence=0.505,
        min_win_rate_median=0.52,
    )
    assert failing["status"] == "FAIL"


class _WeakModel:
    def predict_proba_from_features(self, raw_features):
        return {"fold": 0.80, "call": 0.05, "check": 0.05, "bet": 0.05, "raise": 0.05}

    def predict_from_features(self, raw_features):
        probabilities = self.predict_proba_from_features(raw_features)
        return "fold", probabilities


def test_ev_calibration_can_override_low_value_fold() -> None:
    features = {
        "strength_proxy": 0.9,
        "pot": 10.0,
        "to_call": 1.0,
        "stack": 100.0,
        "min_raise": 4.0,
        "street_index": 0.0,
    }
    probabilities = ev_calibrated_probabilities(features, _WeakModel().predict_proba_from_features(features))
    assert max(probabilities, key=probabilities.get) in {"call", "raise"}


def test_ev_calibrated_policy_exposes_model_interface() -> None:
    features = {
        "strength_proxy": 0.9,
        "pot": 10.0,
        "to_call": 1.0,
        "stack": 100.0,
        "min_raise": 4.0,
        "street_index": 0.0,
    }
    policy = EVCalibratedPolicy(_WeakModel())
    action, probabilities = policy.predict_from_features(features)
    assert action in probabilities
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9
