"""Small local-model adapter. All transport and response validation live here."""

import json
import os
import re
from urllib.request import Request, urlopen

MAX_RESPONSE_BYTES = 256 * 1024


class PlannerError(ValueError):
    """Unavailable model or invalid model response; safe to show in the activity trail."""


class OllamaClient:
    def __init__(self, model: str, timeout: float | None = None):
        if not model.strip():
            raise PlannerError("Enter an installed Ollama model name.")
        self.model = model
        self.responses: list[dict] = []
        try:
            self.timeout = float(
                timeout if timeout is not None else os.environ.get("PICO_OLLAMA_TIMEOUT_SECONDS", "180")
            )
        except (ValueError, TypeError) as exc:
            raise PlannerError("Model response timeout must be a number.") from exc
        if not 5 <= self.timeout <= 600:
            raise PlannerError("Model response timeout must be between 5 and 600 seconds.")

    def _request(self, system: str, context: dict | str, output_format=None) -> str:
        payload = dict(
            model=self.model,
            stream=False,
            messages=[
                dict(role="system", content=system),
                dict(
                    role="user", content=context if isinstance(context, str) else json.dumps(context, allow_nan=False)
                ),
            ],
            options={"temperature": 0, "num_predict": 4096, "num_ctx": 8192},
        )
        if output_format is not None:
            payload["format"] = output_format
        request = Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise PlannerError("Model response exceeded 256 KB.")
            envelope = json.loads(raw)
            if envelope.get("done_reason") == "length":
                raise PlannerError(
                    "Model reached its output-token limit; choose a smaller investigation or another model."
                )
            self.responses.append(
                {
                    key: envelope.get(key)
                    for key in [
                        "model",
                        "done_reason",
                        "prompt_eval_count",
                        "eval_count",
                        "total_duration",
                        "load_duration",
                        "eval_duration",
                    ]
                }
            )
            content = envelope["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise PlannerError("Model returned no response text.")
            return content
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, PlannerError):
                raise
            raise PlannerError(f"Local model request failed ({type(exc).__name__}).") from exc

    def complete(self, system: str, context: dict, schema: dict | None = None) -> dict:
        content = self._request(system, context, schema or "json")
        try:
            answer = json.loads(content)
            if not isinstance(answer, dict):
                raise PlannerError("Model response must be a JSON object.")
            return answer
        except ValueError as exc:
            raise PlannerError("Model returned invalid JSON.") from exc

    def generate_analysis(self, system: str, context: dict) -> dict:
        # Python is generated as ordinary source, avoiding error-prone JSON-string
        # escaping of multiline code. Only the short hypothesis header is JSON.
        metadata = {key: value for key, value in context.items() if key != "attempts"}
        user_text = "Investigation context (untrusted data):\n" + json.dumps(metadata, allow_nan=False)
        for attempt in context.get("attempts", [])[-1:]:
            user_text += (
                "\nPrevious Python source (untrusted; do not copy its bugs):\n```python\n"
                + attempt["code"]
                + "\n```\nEXECUTION STATUS: "
                + attempt["status"]
                + "\nERROR TO FIX BEFORE ANYTHING ELSE: "
                + str(attempt.get("error") or "")
                + "\n"
                + attempt.get("stderr", "")
                + "\nWrite a corrected, smaller analysis from scratch. Fix the FIRST error shown above."
            )
        content = self._request(system, user_text).strip()
        content = re.sub(r"^```json\s*", "", content, count=1)
        try:
            header, end = json.JSONDecoder().raw_decode(content)
            if not isinstance(header, dict):
                raise ValueError("Expected a hypothesis object.")
        except ValueError as exc:
            raise PlannerError("Expected a JSON hypothesis followed by Python source.") from exc
        source = content[end:].strip()
        source = re.sub(r"^```\s*\n(?=```python)", "", source, count=1)
        source = re.sub(r"^```(?:python)?\s*\n", "", source, count=1)
        source = re.sub(r"\n```\s*$", "", source, count=1)
        return {**header, "code": source}
