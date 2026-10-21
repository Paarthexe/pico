"""Small local-model adapter. All transport and response validation live here."""

import json
import os
from urllib.request import Request, urlopen

MAX_RESPONSE_BYTES = 256 * 1024


class PlannerError(ValueError):
    """Unavailable model or invalid model response; safe to show in the activity trail."""


class OllamaClient:
    def __init__(self, model: str, timeout: float | None = None):
        if not model.strip():
            raise PlannerError("Enter an installed Ollama model name.")
        self.model = model
        try:
            self.timeout = float(
                timeout if timeout is not None else os.environ.get("PICO_OLLAMA_TIMEOUT_SECONDS", "180")
            )
        except (ValueError, TypeError) as exc:
            raise PlannerError("Model response timeout must be a number.") from exc
        if not 5 <= self.timeout <= 600:
            raise PlannerError("Model response timeout must be between 5 and 600 seconds.")

    def complete(self, system: str, context: dict, schema: dict | None = None) -> dict:
        payload = dict(
            model=self.model,
            stream=False,
            format=schema or "json",
            messages=[
                dict(role="system", content=system),
                dict(role="user", content=json.dumps(context, allow_nan=False)),
            ],
            options={"temperature": 0, "num_predict": 4096, "num_ctx": 8192},
        )
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
            answer = json.loads(envelope["message"]["content"])
            if not isinstance(answer, dict):
                raise PlannerError("Model response must be a JSON object.")
            return answer
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, PlannerError):
                raise
            raise PlannerError(f"Local model request failed ({type(exc).__name__}).") from exc
