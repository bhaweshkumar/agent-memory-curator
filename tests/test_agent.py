import agent
from memory_store import Memory


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