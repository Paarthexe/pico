"""Portable reports and safe replay: generated source only executes through Docker."""

import json
import zipfile
from dataclasses import asdict
from io import BytesIO

from pico.models import Evidence, Investigation
from pico.plotting import charts_for  # Compatibility public import.
from pico.plotting import generate_plot as generate_plot


def replay_code(evidence: Evidence) -> str:
    if evidence.execution is None:
        return evidence.code
    return (
        "# Generated Python is passed to Docker, never executed on this host.\n"
        "from pathlib import Path\nfrom pico.data import load_csv\n"
        "from pico.sandbox import DockerSandbox\n"
        'dataset = Path("dataset.csv").read_bytes()\n'
        f"source = {evidence.code!r}\n"
        f"result = DockerSandbox().execute(source, dataset, len(load_csv(dataset)), {evidence.id!r})\n"
        'if result.status != "ok":\n    raise RuntimeError(result.error)\n'
        "print(result.result)\n"
    )


def markdown_report(run: Investigation) -> str:
    lines = [
        "# Pico investigation",
        "",
        f"**Question:** {run.question or 'Explore associations with the selected target.'}",
        f"**Target:** {run.target}",
        f"**Planner:** {run.mode}",
        f"**Engine:** {run.engine}",
        f"**Dataset:** {run.profile['rows']:,} rows, {run.profile['columns']} columns",
        "",
        "All results are exploratory associations. Adaptive selection and multiple comparisons "
        "can exaggerate apparent signals. Validate on independent data before acting.",
        "",
    ]
    if not run.evidence:
        lines += ["No eligible analysis produced evidence.", ""]
    if run.multiplicity:
        lines += [
            "## Statistical review",
            "",
            f"Holm adjustment across {run.multiplicity['family_size']} reported tests.",
            run.multiplicity["scope"],
            run.multiplicity["limitation"],
            "",
        ]
    if run.dataset_warnings:
        lines += ["## Data quality and possible leakage", "", *[f"- {w['message']}" for w in run.dataset_warnings], ""]
    for e in run.evidence:
        lines += [
            f"## {e.id}: {e.hypothesis['x']} and {e.hypothesis['y']}",
            "",
            e.conclusion,
            "",
            f"Usable observations: {e.n}.",
            f"Raw p-value: {e.p_value:.4g}." if e.p_value is not None else "No pooled p-value reported.",
            "",
        ]
        if e.execution:
            lines += [
                f"Execution: {e.execution.status}, {e.execution.duration_seconds:.2f}s.",
                f"Worker image: `{e.execution.image}`.",
                "",
            ]
        if e.method:
            lines += [f"Method: {e.method}", ""]
        if e.review:
            if e.review.numerical_summary:
                lines += [e.review.numerical_summary, ""]
            lines += [
                f"Review status: {e.review.status}.",
                f"Holm-adjusted primary p-value: {e.review.adjusted_p_value:.4g}."
                if e.review.adjusted_p_value is not None
                else "No adjusted primary p-value.",
                "",
            ]
            if e.review.original_conclusion != e.conclusion:
                lines += ["Original interpretation (not endorsed):", e.review.original_conclusion, ""]
            lines += [*[f"- {issue['message']}" for issue in e.review.issues], ""]
        for name, _ in charts_for(e):
            lines += [f"![{e.id} analysis chart](charts/{name})", ""]
        lines += [
            *[f"- {c}" for c in e.limitations],
            "",
            "Replay through the installed Pico execution interface:",
            "",
            "```python",
            replay_code(e),
            "```",
            "",
        ]
    failures = [e for e in run.executions if e.status != "ok"]
    if failures:
        lines += ["## Failed execution attempts", ""]
        lines += [f"- {e.id} ({e.status}): {e.error}" for e in failures]
        lines += [""]
    lines += [
        "## Investigation log",
        "",
        *[f"- **{a['stage']}:** {a['message']}" for a in run.activity],
        "",
        "## Suggested next steps",
        "",
        "Confirm candidate associations on held-out data, inspect measurement "
        "quality and possible leakage, and consult domain context before interpreting mechanisms.",
        "",
        f"Stopped: {run.stop_reason}",
    ]
    return "\n".join(lines)


def notebook(run: Investigation) -> str:
    cells = [
        dict(
            cell_type="markdown",
            metadata={},
            source=[
                "# Pico reproducible investigation\n",
                "Install pico-eda and put the original CSV beside this notebook as dataset.csv.\n",
                "Generated analyses require Docker and the built pico-sandbox:1 image.\n",
                "These are exploratory associations, not causal estimates.",
            ],
        )
    ]
    for e in run.evidence:
        cells.append(dict(cell_type="markdown", metadata={}, source=[f"## {e.id}\n", e.conclusion]))
        cells.append(
            dict(
                cell_type="code",
                metadata={},
                execution_count=None,
                outputs=[],
                source=replay_code(e).splitlines(keepends=True),
            )
        )
    return json.dumps(
        dict(
            cells=cells,
            metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            nbformat=4,
            nbformat_minor=4,
        ),
        indent=2,
    )


EXPORT_README = """# Pico investigation bundle

Place the original CSV beside this file as dataset.csv. Install pico-eda.
The notebook and replay/ scripts use the appropriate execution engine.

IMPORTANT: files under code/ and attempts/ can contain model-generated Python.
Do not execute them directly on your host. replay/ scripts send generated code
through Pico's restricted Docker worker. Build pico-sandbox:1 first.

The original dataset is excluded, but evidence, logs, code, and charts can include
sensitive dataset values. Model-generated statistical claims require review.
The original worker image digest is recorded in evidence.json for provenance.
"""


def export_bundle(run: Investigation) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", EXPORT_README)
        archive.writestr("report.md", markdown_report(run))
        archive.writestr("evidence.json", json.dumps(asdict(run), indent=2, allow_nan=False))
        archive.writestr("investigation.ipynb", notebook(run))
        for e in run.evidence:
            for name, png in charts_for(e):
                archive.writestr(f"charts/{name}", png)
            archive.writestr(
                f"code/{e.id}.py",
                "# Untrusted generated source: use replay/ to execute safely.\n" + e.code if e.execution else e.code,
            )
            archive.writestr(f"replay/{e.id}.py", replay_code(e))
        for record in run.executions:
            archive.writestr(f"attempts/{record.id}.json", json.dumps(asdict(record), indent=2, allow_nan=False))
    return output.getvalue()
