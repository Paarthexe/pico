# Pico

**Pattern Investigation and Computational Observation agent**

Pico is a hypothesis-driven CSV analyst. It profiles an unfamiliar dataset, selects an investigation, evaluates executed evidence, and uses the result to choose a follow-up. Every finding includes a chart, sample size, caveats, and replay code.

## Run locally

Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
streamlit run app.py
```

Open http://localhost:8501. The synthetic customer dataset is enabled initially; upload a CSV and disable the demo toggle to analyze your own data. CSV limits: UTF-8, 20 MB, 100,000 rows, 100 columns.

## Try the investigative loop

1. Keep the demo dataset and select `churn` as the outcome.
2. Use the five-analysis budget and start an investigation.
3. Open the investigation trail. An income/churn association prompts Pico to compare the relationship within purchase-frequency strata.
4. Compare the aggregate and stratified evidence. The synthetic data was constructed with purchase frequency affecting churn and income associated with purchase frequency. The stratified comparison is descriptive and does not establish a causal mechanism.
5. Download the investigation bundle for the report, charts, JSON evidence, replay scripts, and Jupyter notebook.

The default planner is deterministic and works without a model or credentials. It prioritizes numeric predictors and generates a follow-up when an executed categorical association has Cramér’s V of at least 0.1. It remembers tested hypotheses, enforces a budget of at most five attempts (including failures), and skips invalid analyses with a visible reason. The default rules use the selected outcome; arbitrary question text does not change their ranking.

## Optional local LLM planner

Install and run [Ollama](https://docs.ollama.com/), pull a model of your choice, select **Ollama local model**, and enter its installed model name. Pico calls the local [chat API](https://docs.ollama.com/api/chat) at `127.0.0.1:11434`, requests JSON, and validates the selected hypothesis index. The model chooses from eligible tools using the question, schema, aggregate profile, and prior aggregate evidence. Raw CSV rows are not sent to the model. Invalid responses or connection failures fall back to the local rules and appear in the trail.

The request has a 45-second network timeout per planning call. This is not a hard generation-time budget on the model server. The live Ollama integration requires a running local model; automated tests cover the request/response contract and fallback behavior.

## Analysis tools and statistical limits

- CSV profiling: inferred numeric/categorical types, missingness, duplicates, constants, distinct counts, numeric quartiles/IQR outliers, and largest category share.
- Numeric/numeric: Spearman correlation.
- Categorical outcome: quartile/category contingency tables, chi-square, and Cramér’s V.
- Numeric outcome by category: Kruskal–Wallis and descriptive group summaries.
- Missingness versus categorical outcome.
- Follow-up: within-stratum category comparisons with weighted Cramér’s V; no pooled p-value is claimed.

P-values are raw and exploratory. Adaptive hypothesis selection, multiple comparisons, sparse expected cells, missing observations, potential identifiers/leakage, and dataset-specific measurement issues limit interpretation. The app reports sparse-cell and small-sample warnings. It does not infer causation, identify time-series semantics, or fit multivariable causal models. Validate candidate findings on independent data.

## Execution boundary

This MVP deliberately uses a fixed catalog of trusted statistical Python tools. The model can select tools but cannot submit arbitrary Python. There is no `eval`, `exec`, shell execution, or model-generated code execution in the application. Replay files call those same installed tools.

**Free-form generated Python and an isolated code-execution sandbox are not implemented.** If adding them, use a separate locked-down worker/container with network disabled, bounded resources, a read-only dataset mount, and a disposable output directory. Do not execute generated Python directly in the Streamlit host.

Uploads live in application memory/session state; there is no database or deliberate upload persistence. The export excludes the original CSV, but includes evidence tables and up to 1,000 plotted pairs for numeric correlations, which can contain dataset values. Treat exported bundles as sensitive as their source data. This is a local MVP without production authentication or multi-user isolation.

## Reproducibility and layout

```text
app.py                    Streamlit upload, health, evidence, and report UI
src/pico/data.py          Ingestion, profiling, seeded synthetic dataset
src/pico/analysis.py      Validated hypotheses and trusted analysis tools
src/pico/agent.py         Plan → execute → evaluate → replan loop
src/pico/report.py        Charts, Markdown, ZIP, and notebook exports
tests/                    Statistical, planner, export, and UI regression tests
```

To replay an export, keep the original CSV as `dataset.csv`, install this project, and run its scripts or notebook. No CSV is bundled automatically. Rule-based demo runs are reproducible; LLM selections depend on the local model. Project design tokens and pending design decisions are recorded in `.tastemaker/`; no personal taste profile was modified.

## Validate

```sh
pytest -q
```

Tests cover ingestion limits, missingness, known correlations, evidence-driven replanning, budgets, model fallback, exports, and Streamlit's complete demo interaction. CI runs the same suite on Python 3.11 and 3.14. Browser screenshots require a connected browser; the initial implementation was validated through Streamlit AppTest, with no claim of browser visual QA.
