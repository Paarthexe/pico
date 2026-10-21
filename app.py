"""Run with: streamlit run app.py"""
import hashlib
import pandas as pd
import streamlit as st
from pico.agent import choose_target, investigate
from pico.data import demo_dataset, load_csv, profile_dataset
from pico.report import export_bundle, generate_plot, markdown_report

st.set_page_config(page_title='Pico | Investigate your data', page_icon='◈', layout='wide')
st.markdown('''<style>
:root { --ink:#121c22; --paper:#f0f9fe; --surface:#e6eff4;
 --primary:#017ca8; --accent:#516fb4; --border:#d6dfe4; --white:#ffffff;
 --radius:6px; --font: "IBM Plex Sans", system-ui, sans-serif; }
html, body { overflow-x:clip; }
.stApp { font-family:var(--font); }
h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
h1 { font-size:clamp(2rem,4vw,3.5rem)!important; line-height:1.1!important; }
[data-testid="stMainBlockContainer"] { max-width:1440px; padding-top:3rem; }
[data-testid="stSidebar"] { border-right:1px solid var(--border); }
[data-testid="stMetric"] { border-top:1px solid var(--border); padding-top:1rem; }
button { border-radius:var(--radius)!important; transition:opacity 120ms ease-out!important; }
button:active { opacity:.8; }
button:focus-visible, input:focus-visible, textarea:focus-visible {
 outline:2px solid var(--primary)!important; outline-offset:3px; }
button:disabled { opacity:.5; cursor:not-allowed; }
@media(prefers-reduced-motion:reduce) { button { transition:none!important; } }
</style>''', unsafe_allow_html=True)

with st.sidebar:
    st.title('◈ pico')
    st.caption('Pattern Investigation and\nComputational Observation')
    st.divider()
    st.subheader('Your dataset')
    uploaded = st.file_uploader('Upload a CSV', type=['csv'], help='UTF-8 CSV, up to 20 MB, 100,000 rows and 100 columns.')
    use_demo = st.toggle('Use synthetic customer demo', value=uploaded is None)
    st.divider()
    st.subheader('Investigation settings')
    budget = st.slider('Analysis budget', 1, 5, 5)
    planner = st.selectbox('Planner', ['Local adaptive rules', 'Ollama local model'])
    model = st.text_input('Ollama model', value='qwen3:8b') if planner.startswith('Ollama') else None
    if model is not None:
        st.caption('Requires Ollama running locally. Schema and aggregate results are sent to localhost; raw rows are not sent.')
    st.caption('Every finding includes executed evidence. All comparisons are exploratory.')

st.caption('PICO / INVESTIGATION WORKSPACE')
st.title('Follow the evidence.')
st.write('Start with an unfamiliar dataset. Let each result shape the next question.')

if uploaded is None and not use_demo:
    st.info('Upload a CSV in the sidebar, or turn on the synthetic demo to explore Pico.')
    st.stop()
try:
    if uploaded is not None and not use_demo:
        raw = uploaded.getvalue()
        df = load_csv(raw)
        filename = uploaded.name
    else:
        df = demo_dataset()
        raw = df.to_csv(index=False).encode()
        filename = 'synthetic_customers.csv'
except ValueError as exc:
    st.error(str(exc))
    st.stop()

fingerprint = hashlib.sha256(raw).hexdigest()
if st.session_state.get('dataset_id') != fingerprint:
    st.session_state.dataset_id = fingerprint
    st.session_state.pop('run', None)
    st.session_state.pop('bundle', None)
    st.session_state.pop('target', None)
profile = profile_dataset(df)
st.write(f'**{filename}**' + (' · Synthetic demonstration data' if use_demo else ''))
metrics = st.columns(4)
for col, label, value in zip(metrics, ['Rows', 'Columns', 'Missing cells', 'Duplicate rows'],
                             [f'{len(df):,}', len(df.columns), f'{profile["missing_percent"]:.1f}%', profile['duplicates']]):
    col.metric(label, value)

with st.expander('Inspect dataset and column health'):
    st.dataframe(df.head(100), hide_index=True, width='stretch')
    st.dataframe(pd.DataFrame(profile['schema']), hide_index=True, width='stretch')
    st.download_button('Download dataset', raw, file_name='dataset.csv', mime='text/csv')

st.divider()
left, right = st.columns([3, 1])
with left:
    question = st.text_input('What would you like to investigate?',
                             placeholder='For example: which patterns are associated with churn?')
with right:
    target = st.selectbox('Outcome to explore', list(df.columns),
                          index=list(df.columns).index(choose_target(df)), key='target')
if planner == 'Local adaptive rules' and question:
    st.caption('Local rules use the selected outcome. Choose Ollama for question-aware hypothesis selection.')
if st.button('Start investigation', type='primary', disabled=df[target].nunique() < 2):
    st.session_state.pop('run', None)
    st.session_state.pop('bundle', None)
    with st.status('Pico is investigating…', expanded=True) as status:
        def update(stage, message):
            st.write(f'**{stage}** · {message}')
        try:
            run = investigate(df, target, question, budget, model, on_event=update)
            st.session_state.run = run
            st.session_state.bundle = export_bundle(run)
            status.update(label=f'Investigation complete · {len(run.evidence)} analyses', state='complete', expanded=False)
        except Exception as exc:
            status.update(label='Investigation could not finish', state='error')
            st.error(f'{type(exc).__name__}: {exc}')

run = st.session_state.get('run')
if run is None:
    st.info('Ready to investigate. Pico can revise its plan when an association warrants a deeper comparison.')
    st.stop()
if (run.target, run.question, run.budget) != (target, question, budget):
    st.warning('Settings have changed. Results below belong to the previous investigation; start a new run to update them.')
st.caption(f'Completed using {run.mode} · Outcome: {run.target} · {run.stop_reason}')
findings, activity, report = st.tabs(['Findings & evidence', 'Investigation trail', 'Report & export'])
with findings:
    st.warning('Exploratory associations only. Adaptive selection, multiple comparisons, missingness, and possible leakage limit interpretation.')
    if not run.evidence:
        st.info('No analysis produced usable evidence. Inspect column health or choose another outcome.')
    for evidence in run.evidence:
        with st.container(border=True):
            st.subheader(f'{evidence.id} · {evidence.hypothesis["x"]} → {evidence.hypothesis["y"]}')
            st.write(evidence.conclusion)
            st.caption(f'{evidence.n:,} usable observations · ' +
                       (f'Raw p = {evidence.p_value:.4g}' if evidence.p_value is not None else 'Descriptive comparison; no pooled p-value'))
            st.image(generate_plot(evidence))
            with st.expander('Evidence, limitations, and replay code'):
                st.dataframe(pd.DataFrame(evidence.table), hide_index=True, width='stretch')
                for caveat in evidence.limitations:
                    st.write(f'• {caveat}')
                st.code(evidence.code, language='python')
with activity:
    for i, event in enumerate(run.activity, 1):
        st.write(f'**{i:02} / {event["stage"]}**')
        st.write(event['message'])
with report:
    st.download_button('Download investigation bundle', st.session_state.get('bundle') or export_bundle(run),
                       file_name='pico-investigation.zip', mime='application/zip')
    st.caption('Includes Markdown report, charts, JSON evidence, replay scripts, and a Jupyter notebook. Your original CSV is excluded.')
    # Keep user-provided column names/question out of raw HTML rendering.
    st.text(markdown_report(run))
