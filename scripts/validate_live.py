"""Opt-in end-to-end validation using a real local Ollama model and Docker."""

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlopen

from pico.agent import investigate
from pico.data import demo_dataset
from pico.ollama import OllamaClient
from pico.planning import PythonPlanner
from pico.report import export_bundle
from pico.sandbox import DockerSandbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("/tmp/pico-live-validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        available = json.loads(response.read())["models"]
    model = next((m for m in available if m["name"] == args.model), None)
    if model is None:
        raise SystemExit(f"Pull {args.model} into Ollama before running live validation.")
    client = OllamaClient(args.model)

    class RecordingSandbox(DockerSandbox):
        def execute(self, code, dataset, dataset_rows, execution_id="A01"):
            record = super().execute(code, dataset, dataset_rows, execution_id)
            (args.output / f"{execution_id}.json").write_text(json.dumps(asdict(record), indent=2, allow_nan=False))
            if record.error:
                print(f"{execution_id} error detail: {record.stderr[-2000:] or record.error}", flush=True)
            return record

    run = investigate(
        demo_dataset(),
        "churn",
        question="Investigate income and churn, then examine whether purchase frequency explains the association.",
        budget=args.budget,
        model=args.model,
        engine="python",
        planner=PythonPlanner(args.model, client),
        executor=RecordingSandbox(),
        on_event=lambda stage, message: print(f"{stage}: {message}", flush=True),
    )
    (args.output / "investigation.json").write_text(json.dumps(asdict(run), indent=2, allow_nan=False))
    (args.output / "investigation.zip").write_bytes(export_bundle(run))
    summary = dict(
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        model_responses=client.responses,
        model=args.model,
        model_digest=model["digest"],
        elapsed_seconds=round(time.monotonic() - started, 2),
        engine=run.engine,
        attempts=len(run.executions),
        findings=len(run.evidence),
        stop_reason=run.stop_reason,
        executions=[
            dict(
                id=e.id,
                status=e.status,
                error=e.error,
                duration=e.duration_seconds,
                image=e.image,
                charts=[a.name for a in e.artifacts],
            )
            for e in run.executions
        ],
        conclusions=[e.conclusion for e in run.evidence],
        activity=run.activity,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if len(run.evidence) < 2:
        raise SystemExit("Live validation did not produce at least two executed findings; inspect its saved evidence.")


if __name__ == "__main__":
    main()
