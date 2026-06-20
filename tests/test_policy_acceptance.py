from poker_agent.policy_acceptance import (
    DeploymentGatedPolicy,
    EVCalibratedPolicy,
    aggregate_simulations,
    ev_calibrated_probabilities,
    evaluate_timing_bet_size_likeness,
    extract_human_behavior_proxies,
    extract_model_behavior_proxies,
    holdem_aware_probabilities,
    jensen_shannon_divergence,
    normalize_distribution,
    total_variation_distance,
)
from poker_agent.schemas import PredictionRequest


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


def test_deployment_gated_policy_preserves_reconstructed_context_predictions() -> None:
    features = {
        "strength_proxy": 0.9,
        "pot": 10.0,
        "to_call": 1.0,
        "stack": 100.0,
        "min_raise": 4.0,
        "street_index": 0.0,
        "call_price_ratio": 0.1,
        "hero_commitment_ratio": 0.0,
        "players_acted_ratio": 0.5,
        "street_action_count": 2.0,
        "street_aggression_ratio": 0.5,
        "table_commitment_pressure": 0.2,
    }
    action, probabilities = DeploymentGatedPolicy(_WeakModel()).predict_from_features(features)
    assert action == "fold"
    assert probabilities["fold"] == 0.80


def test_deployment_gated_policy_uses_ev_overlay_for_ood_context() -> None:
    features = {
        "strength_proxy": 0.9,
        "pot": 10.0,
        "to_call": 1.0,
        "stack": 100.0,
        "min_raise": 4.0,
        "street_index": 0.0,
    }
    action, probabilities = DeploymentGatedPolicy(_WeakModel()).predict_from_features(features)
    assert action in {"call", "raise"}
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9


def test_holdem_aware_probabilities_use_postflop_card_texture() -> None:
    weak_facing_bet = {
        "street_index": 1.0,
        "strength_proxy": 0.30,
        "made_hand_score": 0.0,
        "straight_draw_score": 0.0,
        "flush_draw_pressure": 0.0,
        "to_call": 4.0,
    }
    strong_no_bet = {
        "street_index": 2.0,
        "strength_proxy": 0.50,
        "made_hand_score": 0.78,
        "straight_draw_score": 0.0,
        "flush_draw_pressure": 0.0,
        "to_call": 0.0,
    }

    weak_probabilities = holdem_aware_probabilities(weak_facing_bet)
    strong_probabilities = holdem_aware_probabilities(strong_no_bet)

    assert max(weak_probabilities, key=weak_probabilities.get) == "fold"
    assert max(strong_probabilities, key=strong_probabilities.get) in {
        "check",
        "raise",
    }


def test_deployment_gated_policy_exposes_request_aware_self_play_selector() -> None:
    request = PredictionRequest(
        position="BTN",
        street="flop",
        hole_cards=["AS", "KD"],
        board_cards=["2C", "7D", "9H"],
        pot=10.0,
        to_call=8.0,
        stack=100.0,
        min_raise=16.0,
        player_count=6,
    )

    action, probabilities = DeploymentGatedPolicy(_WeakModel()).predict_request(request)

    assert action == "fold"
    assert probabilities["fold"] > probabilities["call"]


class _RaiseModel:
    def predict_from_features(self, raw_features):
        return "raise", {"fold": 0.05, "call": 0.10, "check": 0.05, "bet": 0.20, "raise": 0.60}


def test_human_behavior_proxy_extracts_timing_and_bet_size(tmp_path) -> None:
    (tmp_path / "actions.csv").write_text(
        "\n".join(
            [
                "hand_id,hand_index,local_hand_index,source_file,frame_id,player_position,player_nickname,action,street",
                "h1,1,1,s,1,SB,,fold,preflop",
                "h1,1,1,s,6,BB,,call,preflop",
                "h1,1,1,s,42,BTN,,raise,preflop",
                "h1,1,1,s,50,BTN,,ante,preflop",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "stack_events.csv").write_text(
        "\n".join(
                [
                    "hand_id,hand_index,local_hand_index,source_file,frame_id,player_position,event,stack,diff,stack_after_event",
                    "h1,1,1,s,42,BTN,update_stack,100,,100",
                    "h1,1,1,s,43,BTN,update_stack,96,,96",
                    "h1,1,1,s,44,BTN,update_stack,80,,80",
            ]
        ),
        encoding="utf-8",
    )

    payload = extract_human_behavior_proxies(tmp_path, max_rows=100)

    assert payload["samples"]["timing_samples"] == 2
    assert payload["timing_counts"]["short"] == 1
    assert payload["timing_counts"]["long"] == 1
    assert payload["samples"]["bet_size_samples"] == 2
    assert payload["bet_size_counts"]["small"] == 1
    assert payload["bet_size_counts"]["large"] == 1


def test_model_behavior_proxy_uses_action_plan_outputs() -> None:
    request = PredictionRequest(
        position="BTN",
        street="preflop",
        hole_cards=["AS", "KD"],
        board_cards=[],
        pot=10.0,
        to_call=2.0,
        stack=100.0,
        min_raise=4.0,
        player_count=6,
    )
    records = [(request, {"strength_proxy": 0.9, "pot": 10.0, "to_call": 2.0, "stack": 100.0}, "raise")]

    payload = extract_model_behavior_proxies(_RaiseModel(), records)

    assert payload["samples"]["timing_samples"] == 1
    assert payload["samples"]["positive_bet_size_samples"] == 1


def test_timing_bet_size_likeness_returns_measurable_status(tmp_path) -> None:
    (tmp_path / "actions.csv").write_text(
        "\n".join(
            [
                "hand_id,hand_index,local_hand_index,source_file,frame_id,player_position,player_nickname,action,street",
                "h1,1,1,s,1,SB,,fold,preflop",
                "h1,1,1,s,20,BB,,call,preflop",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "stack_events.csv").write_text(
        "\n".join(
                [
                    "hand_id,hand_index,local_hand_index,source_file,frame_id,player_position,event,stack,diff,stack_after_event",
                    "h1,1,1,s,20,BB,update_stack,100,,100",
                    "h1,1,1,s,21,BB,update_stack,90,,90",
                ]
            ),
        encoding="utf-8",
    )
    request = PredictionRequest(
        position="BTN",
        street="preflop",
        hole_cards=["AS", "KD"],
        board_cards=[],
        pot=10.0,
        to_call=2.0,
        stack=100.0,
        min_raise=4.0,
        player_count=6,
    )
    records = [(request, {"strength_proxy": 0.9, "pot": 10.0, "to_call": 2.0, "stack": 100.0}, "raise")]

    payload = evaluate_timing_bet_size_likeness(
        _RaiseModel(),
        records,
        tmp_path,
        max_behavior_rows=100,
        max_model_examples=10,
        min_behavior_samples=1,
    )

    assert payload["status"] in {"PASS", "FAIL"}
    assert payload["timing"]["status"] in {"PASS", "FAIL"}
    assert payload["bet_size"]["status"] in {"PASS", "FAIL"}
