"""UI composition and state transitions; domain work is delegated to the agent."""

import hashlib

import streamlit as st

from pico.agent import investigate
from pico.data import demo_dataset, load_csv, profile_dataset
from pico.planning import choose_target
from pico.report import export_bundle
from pico.ui.components import apply_style, dataset_health, results, sidebar
from pico.ui.state import clear_results, select_dataset, settings_signature


def main() -> None:
    st.set_page_config(page_title="Pico | Investigate your data", page_icon="◈", layout="wide")
    apply_style()
    options = sidebar()
    st.caption("PICO / INVESTIGATION WORKSPACE")
    st.title("Follow the evidence.")
    st.write("Start with an unfamiliar dataset. Let each result shape the next question.")
    if options.uploaded_bytes is None and not options.use_demo:
        st.info("Upload a CSV in the sidebar, or turn on the synthetic demo to explore Pico.")
        return
    try:
        if options.use_demo:
            df = demo_dataset()
            raw, filename = df.to_csv(index=False).encode(), "synthetic_customers.csv"
        else:
            raw, filename = options.uploaded_bytes, options.filename
            df = load_csv(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    select_dataset(st.session_state, hashlib.sha256(raw).hexdigest())
    st.write(f"**{filename}**" + (" · Synthetic demonstration data" if options.use_demo else ""))
    dataset_health(df, profile_dataset(df), raw)
    st.divider()
    left, right = st.columns([3, 1])
    with left:
        question = st.text_input(
            "What would you like to investigate?", placeholder="For example: which patterns are associated with churn?"
        )
    with right:
        target = st.selectbox(
            "Outcome to explore", list(df.columns), index=list(df.columns).index(choose_target(df)), key="target"
        )
    if options.engine == "catalog" and options.model is None and question:
        st.caption("Local rules use the selected outcome. Choose Ollama for question-aware hypothesis selection.")
    signature = settings_signature(target, question, options.budget, options.model, options.engine)
    if st.button("Start investigation", type="primary", disabled=df[target].nunique() < 2):
        clear_results(st.session_state)
        with st.status("Pico is investigating…", expanded=True) as status:

            def update(stage: str, message: str) -> None:
                st.write(f"**{stage}** · {message}")

            try:
                run = investigate(
                    df, target, question, options.budget, options.model, on_event=update, engine=options.engine
                )
                bundle = export_bundle(run)
            except Exception as exc:
                # The UI boundary reports unexpected failures; lower layers only catch
                # errors they can handle. Never substitute host execution after failure.
                status.update(label="Investigation could not finish", state="error")
                st.error(f"{type(exc).__name__}: {exc}")
            else:
                st.session_state.update(run=run, bundle=bundle, run_signature=signature)
                status.update(
                    label=f"Investigation complete · {len(run.evidence)} analyses", state="complete", expanded=False
                )
    if "run" not in st.session_state:
        st.info("Ready to investigate. Pico can revise its plan when an association warrants a deeper comparison.")
        return
    if st.session_state.run_signature != signature:
        st.warning(
            "Settings have changed. Results below belong to the previous investigation; start a new run to update them."
        )
    results(st.session_state.run, st.session_state.bundle)
