"""Reproducible model evaluation on synthetic datasets with explicit ground truth.

Preparing fixtures needs no model. --live invokes Ollama and Docker through the
same investigation path as the app. Reports retain failures; a matched calculation
is separate from an accepted interpretation and from a valid study design.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def fixtures():
    cases = {}
    for name, counts, recipe in [
        ("null", [[40, 40], [40, 40]], "pearson_chi2"),
        ("signal", [[80, 20], [20, 80]], "pearson_chi2"),
        ("sparse", [[1, 9], [8, 2]], "fisher_exact"),
    ]:
        frame = pd.DataFrame(
            [(str(i), str(j)) for i, row in enumerate(counts) for j, n in enumerate(row) for _ in range(n)],
            columns=["x", "y"],
        )
        cases[name] = (frame, recipe)
    missing = cases["signal"][0].copy()
    missing.loc[missing.index[(missing.x == "0") & (missing.y == "1")][:10], "x"] = None
    cases["missingness"] = (missing, "pearson_chi2")
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--cases", nargs="+", choices=list(fixtures()), default=list(fixtures()))
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("/tmp/pico-analytical-evaluation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.cases:
        frame, recipe = fixtures()[name]
        csv = args.output / f"{name}.csv"
        frame.to_csv(csv, index=False)
        entry = dict(case=name, recipe=recipe, rows=len(frame), status="prepared")
        if args.live:
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            question = (
                f"Compare x against y with {recipe}, original categories, pairwise complete cases, no imputation. "
                "Use the independently checkable flat group/category/count table and verification diagnostics. "
                'Use correction=False for Pearson or alternative="two-sided" for Fisher. '
                "Do not claim causality or interpret insufficient evidence as proof of no relationship."
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("validate_live.py")),
                "--dataset",
                str(csv),
                "--target",
                "y",
                "--model",
                args.model,
                "--budget",
                str(args.budget),
                "--minimum-findings",
                "1",
                "--require-verified",
                "--question",
                question,
                "--output",
                str(directory),
            ]
            print(f"Evaluating {name} with {args.model}", flush=True)
            with (directory / "run.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1500, check=False)
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            entry.update(
                status="passed" if process.returncode == 0 else "failed",
                returncode=process.returncode,
                attempts=summary.get("attempts"),
                findings=summary.get("findings"),
                verification=summary.get("verification", []),
                reviews=summary.get("reviews", []),
            )
            print(json.dumps(entry), flush=True)
        results.append(entry)
        (args.output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(entry["status"] == "failed" for entry in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
