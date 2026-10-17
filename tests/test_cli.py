import sys
from types import SimpleNamespace

import pytest

import cli
from decider_client import Decision, StrandsDeciderClient


def test_main_defaults_to_real_decider(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "run_repl", lambda **kwargs: captured.update(kwargs))
    cli.main(["--question", "pytest tests"])
    assert isinstance(captured["decider"], StrandsDeciderClient)
    assert captured["questions"] == ["pytest tests"]


def test_failed_scoring_turn_does_not_call_bedrock(monkeypatch, capsys):
    class UnavailableDecider:
        def score(self, question, context, option):
            raise ConnectionError("server unavailable")

    def unexpected_call(*args):
        pytest.fail("Bedrock must not run when all decisions failed")

    monkeypatch.setitem(sys.modules, "agent", SimpleNamespace(answer_question=unexpected_call))
    cli.run_repl(decider=UnavailableDecider(), questions=["pytest tests"])
    assert "Skipping Bedrock" in capsys.readouterr().out


def test_only_curated_memories_reach_bedrock(monkeypatch, capsys):
    class TestDecider:
        def score(self, question, context, option):
            return Decision(option, 0.9 if "pytest" in context else 0.1)

    calls = []

    def answer(question, memories):
        calls.append((question, memories))
        return "Use pytest."

    monkeypatch.setitem(sys.modules, "agent", SimpleNamespace(answer_question=answer))
    cli.run_repl(decider=TestDecider(), questions=["pytest tests"])
    assert len(calls) == 1
    assert [memory.id for memory in calls[0][1]] == ["mem-001"]
    assert "Use pytest." in capsys.readouterr().out


def test_bedrock_failure_does_not_stop_next_question(monkeypatch, capsys):
    class TestDecider:
        def score(self, question, context, option):
            return Decision(option, 0.0)

    calls = []

    def answer(question, memories):
        calls.append(question)
        if len(calls) == 1:
            raise RuntimeError("AWS unavailable")
        assert memories == []
        return "Second answer"

    monkeypatch.setitem(sys.modules, "agent", SimpleNamespace(answer_question=answer))
    cli.run_repl(decider=TestDecider(), questions=["first", "second"])
    assert calls == ["first", "second"]
    output = capsys.readouterr().out
    assert "Bedrock call failed" in output
    assert "Second answer" in output