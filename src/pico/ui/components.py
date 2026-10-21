"""Focused view functions. Controls and evidence rendering do not run investigations."""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from pico.models import Evidence, Investigation
from pico.plotting import charts_for
from pico.report import markdown_report


@dataclass(frozen=True)
class Options:
    uploaded_bytes: bytes | None
    filename: str
    use_demo: bool
    budget: int
    model: str | None
    engine: str


def apply_style() -> None:
    css = Path(__file__).with_name("styles.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar() -> Options:
    with st.sidebar:
        st.title("◈ pico")
        st.caption("Pattern Investigation and\nComputational Observation")
        st.divider()
        st.subheader("Your dataset")
        uploaded = st.file_uploader(
            "Upload a CSV", type=["csv"], help="UTF-8 CSV, up to 20 MB, 100,000 rows and 100 columns."
        )
        demo = st.toggle("Use synthetic customer demo", value=uploaded is None)
        st.divider()
        st.subheader("Investigation settings")
        budget = st.slider("Analysis budget", 1, 5, 5)
        engine_label = st.selectbox("Analysis engine", ["Catalog tools", "Generated Python (Docker)"])
        engine = "python" if engine_label.startswith("Generated") else "catalog"
        if engine == "python":
            use_model = True
            st.caption(
                "Pico writes new Python analyses, runs them in disposable containers, and learns from results and execution errors."
            )
            st.caption(
                "Requires Docker and the built pico-sandbox:1 image. Each attempt has a 45-second execution limit."
            )
        else:
            use_model = st.selectbox("Planner", ["Local adaptive rules", "Ollama local model"]) == "Ollama local model"
        model = (
            st.text_input("Ollama model", value=os.environ.get("PICO_OLLAMA_MODEL", "qwen2.5-coder:7b"))
            if use_model
            else None
        )
        if use_model:
            st.caption(
                "Requires Ollama running locally. Schema, aggregate findings, and generated execution errors are sent to localhost."
            )
        st.caption("Every finding includes executed evidence. All comparisons are exploratory.")
    return Options(
        uploaded.getvalue() if uploaded else None,
        uploaded.name if uploaded else "synthetic_customers.csv",
        demo,
        budget,
        model,
        engine,
    )


def dataset_health(df: pd.DataFrame, profile: dict, raw: bytes) -> None:
    metrics = st.columns(4)
    values = [f"{len(df):,}", len(df.columns), f"{profile['missing_percent']:.1f}%", profile["duplicates"]]
    for col, label, value in zip(metrics, ["Rows", "Columns", "Missing cells", "Duplicate rows"], values):
        col.metric(label, value)
    with st.expander("Inspect dataset and column health"):
        st.dataframe(df.head(100), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(profile["schema"]), hide_index=True, width="stretch")
        st.download_button("Download dataset", raw, file_name="dataset.csv", mime="text/csv")


def evidence_card(evidence: Evidence) -> None:
    with st.container(border=True):
        st.subheader(f"{evidence.id} · {evidence.hypothesis['x']} → {evidence.hypothesis['y']}")
        st.write(evidence.conclusion)
        st.caption(
            f"{evidence.n:,} usable observations · "
            + (f"Raw p = {evidence.p_value:.4g}" if evidence.p_value is not None else "No pooled p-value")
        )
        if evidence.method:
            st.caption(evidence.method)
        if evidence.review:
            review = evidence.review
            if review.adjusted_p_value is not None:
                st.caption(
                    f"Holm-adjusted p = {review.adjusted_p_value:.4g} · {review.family_size} reported tests · exploratory"
                )
            if review.status == "unsupported":
                st.error(
                    "Interpretation flagged by the statistical review. Inspect the numerical evidence and review notes."
                )
            with st.expander("Statistical review and assumptions"):
                if review.original_conclusion != evidence.conclusion:
                    st.caption("Original interpretation, flagged as unsupported:")
                    st.text(review.original_conclusion)
                for issue in review.issues:
                    st.write(f"• {issue['message']}")
                for assumption in evidence.assumptions:
                    st.write(f"Assumption: {assumption}")
                if evidence.diagnostics:
                    st.json(evidence.diagnostics)
                if review.adjusted_tests:
                    st.dataframe(pd.DataFrame(review.adjusted_tests), hide_index=True, width="stretch")
        for name, data in charts_for(evidence):
            st.image(
                data,
                caption=name if evidence.execution else None,
                alt=f"{evidence.id}: {evidence.hypothesis['x']} and {evidence.hypothesis['y']}. {evidence.conclusion} Numerical values are in the evidence table.",
            )
        with st.expander("Evidence, limitations, and source code"):
            st.dataframe(pd.DataFrame(evidence.table), hide_index=True, width="stretch")
            for caveat in evidence.limitations:
                st.write(f"• {caveat}")
            st.code(evidence.code, language="python")
            if evidence.execution:
                st.caption(
                    f"Executed in Docker · {evidence.execution.duration_seconds:.2f}s · {evidence.execution.image}"
                )
                st.text(evidence.execution.stdout or "No stdout.")
                if evidence.execution.stderr:
                    st.code(evidence.execution.stderr, language="text")


def results(run: Investigation, bundle: bytes) -> None:
    st.caption(f"Completed using {run.mode} · Outcome: {run.target} · {run.stop_reason}")
    findings, activity, report = st.tabs(["Findings & evidence", "Investigation trail", "Report & export"])
    with findings:
        st.warning(
            "Exploratory associations only. Adaptive selection, multiple comparisons, missingness, and possible leakage limit interpretation."
        )
        if run.multiplicity:
            st.caption(
                f"Holm adjustment: {run.multiplicity['family_size']} reported tests. {run.multiplicity['limitation']}"
            )
        if run.dataset_warnings:
            with st.expander("Data quality and possible leakage"):
                for warning in run.dataset_warnings:
                    st.write(f"• {warning['message']}")
        if not run.evidence:
            st.info(
                "No analysis produced usable evidence. Inspect the investigation trail for execution errors or choose another outcome."
            )
        for evidence in run.evidence:
            evidence_card(evidence)
    with activity:
        for index, event in enumerate(run.activity, 1):
            st.write(f"**{index:02} / {event['stage']}**")
            st.write(event["message"])
        for execution in run.executions:
            with st.expander(f"{execution.id} execution · {execution.status} · {execution.duration_seconds:.2f}s"):
                if execution.error:
                    st.error(execution.error)
                st.code(execution.code, language="python")
                st.code(execution.stdout or "No stdout.", language="text")
                st.code(execution.stderr or "No stderr.", language="text")
    with report:
        st.download_button(
            "Download investigation bundle", bundle, file_name="pico-investigation.zip", mime="application/zip"
        )
        st.caption(
            "Includes report, charts, evidence, source, sandbox replay scripts, notebook, and failed attempts. Original CSV excluded."
        )
        # Generated text and dataset names never enter raw HTML rendering.
        st.text(markdown_report(run))
