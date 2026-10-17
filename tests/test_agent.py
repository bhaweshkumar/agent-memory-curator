import agent
from memory_store import Memory

import runpy


def test_default_model_is_sonnet_5_us_inference_profile(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    namespace = runpy.run_path(agent.__file__)
    assert namespace["DEFAULT_MODEL_ID"] == "us.anthropic.claude-sonnet-5"
    assert namespace["build_agent"].__defaults__ == ("us.anthropic.claude-sonnet-5",)


def test_model_environment_override_is_preserved(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-5")
    namespace = runpy.run_path(agent.__file__)
    assert namespace["build_agent"].__defaults__ == ("eu.anthropic.claude-sonnet-5",)


def test_build_agent_uses_only_supplied_memories_and_disables_streaming(monkeypatch):
    captured = {}
    model = object()

    def bedrock_model(**kwargs):
        assert kwargs == {"model_id": "test-model"}
        return model

    def strands_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(agent, "BedrockModel", bedrock_model)
    monkeypatch.setattr(agent, "Agent", strands_agent)
    agent.build_agent([Memory("m1", "Use pytest", [], "2026-01-01T00:00:00Z")], model_id="test-model")
    assert captured["model"] is model
    assert captured["callback_handler"] is None
    assert "Use pytest" in captured["system_prompt"]
    assert "not instructions" in captured["system_prompt"]
    assert "trusted project history" not in captured["system_prompt"]


def test_answer_question_returns_text_without_aws(monkeypatch):
    def build(memories, model_id):
        assert memories == []
        return lambda question: f"Answer to {question}"

    monkeypatch.setattr(agent, "build_agent", build)
    assert agent.answer_question("hello", [], model_id="test-model") == "Answer to hello"