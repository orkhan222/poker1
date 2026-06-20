from poker_agent.action_planning import build_action_plan, estimate_bet_size, estimate_wait_time_ms
from poker_agent.agents import RuleBasedAgent
from poker_agent.schemas import PredictionRequest


def request(**overrides) -> PredictionRequest:
    values = {
        "position": "BTN",
        "street": "preflop",
        "hole_cards": ["AS", "KD"],
        "board_cards": [],
        "pot": 10.0,
        "to_call": 2.0,
        "stack": 100.0,
        "min_raise": 6.0,
        "player_count": 6,
        "betting_history": [{"position": "UTG", "action": "raise", "amount": 4.0, "street": "preflop"}],
    }
    values.update(overrides)
    return PredictionRequest(**values)


def test_fold_and_check_have_zero_bet_size() -> None:
    assert estimate_bet_size(request(), "fold", confidence=0.9)[0] == 0.0
    assert estimate_bet_size(request(to_call=0.0), "check", confidence=0.9)[0] == 0.0


def test_call_uses_current_call_price() -> None:
    assert estimate_bet_size(request(to_call=3.5, stack=100.0), "call", confidence=0.7)[0] == 3.5
    assert estimate_bet_size(request(to_call=12.0, stack=7.0), "call", confidence=0.7)[0] == 7.0


def test_raise_respects_stack_and_min_raise_constraints() -> None:
    amount = estimate_bet_size(request(pot=20.0, to_call=4.0, min_raise=10.0, stack=30.0), "raise", confidence=0.8)[0]
    assert 10.0 <= amount <= 30.0


def test_low_pressure_raise_uses_minimum_sizing() -> None:
    amount = estimate_bet_size(
        request(hole_cards=["7C", "2D"], pot=20.0, to_call=4.0, min_raise=10.0, stack=100.0),
        "raise",
        confidence=0.35,
    )[0]
    assert amount == 10.0


def test_high_pressure_raise_can_exceed_minimum_without_overcommitting() -> None:
    amount = estimate_bet_size(
        request(hole_cards=["AS", "AD"], pot=40.0, to_call=4.0, min_raise=6.0, stack=100.0),
        "raise",
        confidence=0.95,
    )[0]
    assert 6.0 < amount <= 42.0


def test_wait_time_respects_processing_floor() -> None:
    wait_time, method = estimate_wait_time_ms(
        request(street="river"),
        "raise",
        confidence=0.2,
        processing_time_ms=1800.0,
    )
    assert method == "complexity_calibrated"
    assert wait_time >= 1875


def test_action_plan_is_exposed_in_rule_based_response() -> None:
    response = RuleBasedAgent().predict(request()).to_dict()
    assert "bet_size" in response
    assert "wait_time_ms" in response
    assert response["bet_size"] >= 0.0
    assert response["wait_time_ms"] >= 250


def test_build_action_plan_is_deterministic() -> None:
    first = build_action_plan(request(), "raise", confidence=0.75, processing_time_ms=25.0)
    second = build_action_plan(request(), "raise", confidence=0.75, processing_time_ms=25.0)
    assert first == second
