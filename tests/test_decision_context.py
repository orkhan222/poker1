from poker_agent.decision_context import build_contextual_decision_prompt, describe_prompt_profile
from poker_agent.llm_decision import LLMDecisionAgent, HeuristicTextProvider
from poker_agent.schemas import PredictionRequest


def sample_request() -> PredictionRequest:
    return PredictionRequest(
        position="BTN",
        street="preflop",
        hole_cards=["AS", "KD"],
        board_cards=[],
        pot=2.5,
        to_call=1.0,
        stack=100.0,
        min_raise=4.5,
        player_count=6,
        betting_history=[{"position": "UTG", "action": "raise", "amount": 3.0, "street": "preflop"}],
    )


def test_rules_zero_shot_prompt_contains_task_rules_and_schema() -> None:
    prompt = build_contextual_decision_prompt(
        sample_request(),
        allowed_actions=("fold", "call", "check", "bet", "raise"),
        profile="rules_zero_shot",
    )
    assert "Task: classify one poker decision" in prompt
    assert "Poker rules and decision context" in prompt
    assert "Output schema" in prompt
    assert "Examples:" not in prompt
    assert '"action"' in prompt


def test_few_shot_prompt_includes_requested_examples() -> None:
    prompt = build_contextual_decision_prompt(
        sample_request(),
        allowed_actions=("fold", "call", "check", "bet", "raise"),
        profile="rules_few_shot",
        few_shot_count=3,
    )
    assert prompt.count("Example output:") == 3
    assert "Current state:" in prompt


def test_candidate_ranker_prompt_includes_candidates() -> None:
    prompt = build_contextual_decision_prompt(
        sample_request(),
        allowed_actions=("fold", "call", "check", "bet", "raise"),
        profile="candidate_ranker",
        candidates=("fold", "call", "raise"),
    )
    assert "Candidate actions:" in prompt
    assert '"raise"' in prompt
    assert "Do not introduce an action outside the candidate list" in prompt


def test_agent_attaches_prompt_to_response() -> None:
    agent = LLMDecisionAgent(
        provider=HeuristicTextProvider(),
        mode="candidate_ranker",
        prompt_profile="candidate_ranker",
        few_shot_count=2,
    )
    response = agent.predict(sample_request())
    assert response.action in response.probabilities
    assert "Poker rules and decision context" in response.prompt


def test_prompt_profile_description_is_reproducible() -> None:
    description = describe_prompt_profile("rules_few_shot", few_shot_count=5)
    assert description["few_shot_count"] == 5
    assert description["rules_count"] > 0
    assert description["output_schema"]["action"].startswith("fold")
