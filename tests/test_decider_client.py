from decider_client import Decision, MockDeciderClient

import json

import httpx
import pytest


def test_real_client_uses_official_noul_contract():
    from decider_client import StrandsDeciderClient

    def respond(request):
        assert request.url.path == "/v1/systemone"
        payload = json.loads(request.content)
        assert json.loads(payload["state"]) == {"question": "pytest tests", "memory": "Use pytest"}
        assert payload["questions"]["relevance"]["type"] == "noul"
        return httpx.Response(200, json={
            "model": "strands-decider-2B-qwen3.5-v1-2610",
            "answers": {"relevance": {"type": "noul", "noul": 0.85}},
        })

    client = StrandsDeciderClient(http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    assert client.score("pytest tests", "Use pytest", "relevant") == Decision("relevant", 0.85)


@pytest.mark.parametrize("probability", [True, -0.1, 1.1, "0.9", None])
def test_real_client_rejects_invalid_probabilities(probability):
    from decider_client import StrandsDeciderClient

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "model": "strands-decider-2B-qwen3.5-v1-2610",
        "answers": {"relevance": {"type": "noul", "noul": probability}},
    }))
    client = StrandsDeciderClient(http_client=httpx.Client(transport=transport))
    with pytest.raises(ValueError):
        client.score("question", "memory", "relevant")


def test_real_client_rejects_wrong_model():
    from decider_client import StrandsDeciderClient

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "model": "some-other-model",
        "answers": {"relevance": {"type": "noul", "noul": 0.9}},
    }))
    client = StrandsDeciderClient(http_client=httpx.Client(transport=transport))
    with pytest.raises(ValueError, match="model"):
        client.score("question", "memory", "relevant")


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
