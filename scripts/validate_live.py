"""Opt-in end-to-end validation using a real local Ollama model and Docker."""

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from pico.agent import investigate
from pico.data import demo_dataset
from pico.ollama import OllamaClient
from pico.planning import PythonPlanner
from pico.report import export_bundle
from pico.sandbox import DockerSandbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument(
        "--question",
        default="Compare churn rates across income quartiles. Use the executed findings to choose a follow-up analysis involving purchase frequency; report only associations.",
    )
    parser.add_argument(
        "--inject-first-error",
        action="store_true",
        help="Explicitly inject a labeled first-attempt failure to test live repair.",
    )
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--dataset", type=Path, help="Optional evaluation CSV; default is the demo dataset.")
    parser.add_argument("--target", default="churn")
    parser.add_argument("--require-verified", action="store_true", help="Require independently matched findings.")
    parser.add_argument(
        "--minimum-findings", type=int, default=2, help="Required executed findings for this smoke test."
    )
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
            (args.output / f"{execution_id}.generated.py.txt").write_text(code)
            if args.inject_first_error and execution_id == "A01":
                code = "raise NameError('PICO_VALIDATION_INJECTED_FAILURE')\n" + code
            record = super().execute(code, dataset, dataset_rows, execution_id)
            (args.output / f"{execution_id}.json").write_text(json.dumps(asdict(record), indent=2, allow_nan=False))
            if record.error:
                print(f"{execution_id} error detail: {record.stderr[-2000:] or record.error}", flush=True)
            return record

    run = investigate(
        pd.read_csv(args.dataset) if args.dataset else demo_dataset(),
        args.target,
        question=args.question,
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
        fault_injection=args.inject_first_error,
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        model_responses=client.responses,
        question=args.question,
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
        verification=[dict(id=e.id, **e.review.verification) for e in run.evidence if e.review],
        reviews=[
            dict(
                id=e.id,
                status=e.review.status,
                original_conclusion=e.review.original_conclusion,
                issues=e.review.issues,
            )
            for e in run.evidence
            if e.review
        ],
        activity=run.activity,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if len(run.evidence) < args.minimum_findings:
        raise SystemExit(
            f"Live validation did not produce {args.minimum_findings} executed findings; inspect its saved evidence."
        )
    if (
        args.require_verified
        and sum(
            e.review.verification.get("status") == "matched" and e.review.status != "unsupported"
            for e in run.evidence
            if e.review
        )
        < args.minimum_findings
    ):
        raise SystemExit("Live validation needs independently matched findings without blocking review issues.")


if __name__ == "__main__":
    main()
