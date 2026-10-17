from decider_client import Decision, MockDeciderClient


def test_deterministic_scoring_same_inputs_same_confidence():
    decider = MockDeciderClient()

    d1 = decider.score(question="How do I retry a failed call?", context="retryOperation handles retries", option="relevant")
    d2 = decider.score(question="How do I retry a failed call?", context="retryOperation handles retries", option="relevant")

    assert d1 == d2
    assert isinstance(d1, Decision)


def test_high_overlap_scores_higher_than_low_overlap():
    decider = MockDeciderClient()
    question = "How do I retry a failed network call?"

    high_overlap = decider.score(question=question, context="We fixed retry logic for failed network calls.", option="relevant")
    low_overlap = decider.score(question=question, context="This is about unrelated database indexing topics.", option="relevant")

    assert high_overlap.confidence > low_overlap.confidence


def test_confidence_is_clamped_between_zero_and_one():
    decider = MockDeciderClient()
    decision = decider.score(question="retry retry retry", context="retry retry retry", option="relevant")

    assert 0.0 <= decision.confidence <= 1.0


def test_empty_question_or_context_yields_zero_confidence():
    decider = MockDeciderClient()

    assert decider.score(question="", context="something", option="relevant").confidence == 0.0
    assert decider.score(question="something", context="", option="relevant").confidence == 0.0
