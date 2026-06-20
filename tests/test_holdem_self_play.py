from __future__ import annotations

from poker_agent.holdem_self_play import evaluate_seven_cards, run_holdem_self_play


class _AggressiveModel:
    def predict_from_features(self, raw_features):
        probabilities = {
            "fold": 0.04,
            "call": 0.16,
            "check": 0.08,
            "bet": 0.20,
            "raise": 0.52,
        }
        return "raise", probabilities


def test_showdown_evaluator_orders_major_hand_classes() -> None:
    straight_flush = evaluate_seven_cards(["AS", "KS", "QS", "JS", "TS", "2D", "3C"])
    quads = evaluate_seven_cards(["9S", "9H", "9D", "9C", "2S", "3H", "4D"])
    full_house = evaluate_seven_cards(["AH", "AD", "AC", "KH", "KD", "2S", "3S"])

    assert straight_flush > quads
    assert quads > full_house


def test_holdem_self_play_reports_validated_environment() -> None:
    report = run_holdem_self_play(
        _AggressiveModel(),
        seeds=[7],
        hands_per_seed=6,
        player_count=3,
        min_win_rate_high_confidence=0.505,
        min_win_rate_median=0.52,
    )

    assert report["simulation_type"] == "validated_multi_agent_holdem_self_play"
    assert report["environment"]["showdown_evaluator"] == "complete_5_of_7_card_ranker"
    assert report["run_count"] == 1
    assert report["status"] in {"PASS", "FAIL"}
    assert 0.0 <= report["mean_policy_win_rate"] <= 1.0
