import json
from io import BytesIO

import pytest

from pico.ollama import OllamaClient, PlannerError


@pytest.mark.parametrize("wrapper", ["", "```json\n"])
def test_plain_python_generation_preserves_quotes_newlines_and_headers(monkeypatch, wrapper):
    code = 'from pico_tools import load_dataset\ndf = load_dataset()\nprint(f"Rows: {len(df)}")'
    header = json.dumps(dict(hypothesis="Measure rows", x="x", y="outcome"))
    content = wrapper + header + ("\n```" if wrapper else "") + "\n```python\n" + code + "\n```"

    def fake_open(request, timeout):
        payload = json.loads(request.data)
        assert "format" not in payload
        assert "Previous Python source" in payload["messages"][1]["content"]
        assert "FIRST error" in payload["messages"][1]["content"]
        return BytesIO(json.dumps({"message": {"content": content}, "done_reason": "stop"}).encode())

    monkeypatch.setattr("pico.ollama.urlopen", fake_open)
    result = OllamaClient("fixture").generate_analysis(
        "system", dict(attempts=[dict(code="bad", status="error", error="fix", stderr="trace")])
    )
    assert result["code"] == code
    assert result["x"] == "x"


def test_model_limit_and_invalid_output_are_explicit(monkeypatch):
    monkeypatch.setattr(
        "pico.ollama.urlopen", lambda *args, **kwargs: BytesIO(json.dumps(dict(done_reason="length")).encode())
    )
    with pytest.raises(PlannerError, match="output-token limit"):
        OllamaClient("fixture").complete("system", {})
    monkeypatch.setenv("PICO_OLLAMA_TIMEOUT_SECONDS", "0")
    with pytest.raises(PlannerError, match="timeout"):
        OllamaClient("fixture")


def test_successful_analysis_becomes_followup_context_not_a_repair_request(monkeypatch):
    def fake_open(request, timeout):
        message = json.loads(request.data)["messages"][1]["content"]
        assert "FOLLOW-UP" in message
        assert "old completed source" not in message
        assert "ERROR TO FIX" not in message
        content = '{"hypothesis":"New question","x":"x","y":"outcome"}\n```python\nprint(2)\n```'
        return BytesIO(json.dumps({"message": {"content": content}}).encode())

    monkeypatch.setattr("pico.ollama.urlopen", fake_open)
    result = OllamaClient("fixture").generate_analysis(
        "system", dict(attempts=[dict(code="old completed source", status="ok")])
    )
    assert result["code"] == "print(2)"
