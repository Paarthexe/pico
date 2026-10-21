# Pico

**Pattern Investigation and Computational Observation agent**

Pico explores CSV datasets through an observe → hypothesize → execute → evaluate → replan loop. Choose trusted statistical tools or let a local model write new Python analyses and run them in disposable, restricted Docker containers. Findings retain numerical evidence, charts, code, execution logs, and limitations.

## Run locally

Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
streamlit run app.py
```

Open http://localhost:8501. The synthetic customer dataset is enabled initially. For your data, upload a UTF-8 CSV and disable the demo toggle. Limits: 20 MB, 100,000 rows, 100 columns.

## Generated Python with isolated execution

1. Install/start Docker with Linux containers and seccomp support.
2. Build the worker image from the project root:

   ```sh
   docker build -f sandbox/Dockerfile -t pico-sandbox:1 .
   ```

3. Install/start [Ollama](https://docs.ollama.com/) and pull a model you want to use. Enter its installed name in Pico.
4. Select **Generated Python (Docker)** as the analysis engine, choose an outcome, and start an investigation.

Pico asks the model for a hypothesis and Python source, then sends that source to the worker. It can use pandas, NumPy, SciPy, Matplotlib, Seaborn, statsmodels, and scikit-learn, including analyses outside the catalog. Results and execution failures inform the next hypothesis or code repair, within the selected budget of at most five attempts. Successful source is not executed twice. Invalid model output stops with a visible reason; invalid/runtime-failed code remains in the trail and is returned to the model for repair.

Each attempt uses no network, a non-root user, read-only inputs/root, size-limited writable temporary space, no capabilities or new privileges, and CPU, memory, process, file, output, and wall-time limits. The container and its descendants are removed after every attempt. Generated code never runs in the Streamlit host or notebook kernel. Missing Docker/image, unsupported isolation, and failed cleanup stop execution; there is no fallback to host execution.

Docker must be able to bind the canonical host temporary directory. On macOS with Colima, share that directory read-only when starting its VM; see [architecture and setup details](docs/architecture.md#data-and-resource-handling). The worker is intended for a local analyst workspace; stronger multi-user deployment controls are outside this app.

The model receives schema, aggregate profile/findings, previously generated source, and execution error excerpts through the local [Ollama chat API](https://docs.ollama.com/api/chat). Raw CSV rows are not deliberately included in prompts, but result/error text can contain dataset values. The container receives only its dataset snapshot and source. No credentials or general host environment are forwarded.

## Try free-form Python without a model

The executor is also a Python API. This example fits an adjusted logistic regression and saves a chart for the synthetic dataset, entirely inside Docker:

```python
from pathlib import Path
from pico.data import demo_dataset
from pico.sandbox import DockerSandbox

frame = demo_dataset()
result = DockerSandbox().execute(
    Path('examples/adjusted_churn.py').read_text(),
    frame.to_csv(index=False).encode(),
    len(frame),
)
print(result.status, result.result, result.error, result.stderr)
```

`examples/adjusted_churn.py` itself uses container paths and must not be run directly on the host. To author another analysis, read `/input/dataset.csv`, write `/output/result.json` with `n`, `conclusion`, `limitations`, a nonempty scalar evidence `table`, and optional finite `statistic`, `p_value`, `effect`. Save up to four PNGs in `/output`. stdout/stderr are diagnostics, not findings. The full contract is in [the planner prompt](src/pico/planning.py) and [protocol validator](src/pico/sandbox/protocol.py).

## Catalog mode and demo

**Catalog tools** works immediately without Docker or model credentials. Local rules rank candidates and investigate associations; optional Ollama selects hypotheses using your question and prior evidence. Local rules use the selected outcome rather than interpreting arbitrary question text.

Keep the synthetic dataset, select `churn`, and use five attempts. An observed income/churn association leads to a comparison within purchase-frequency strata. The demo was constructed with purchase frequency affecting churn and income associated with purchase frequency. The stratified comparison is descriptive, not a causal estimate.

Catalog analyses include Spearman correlation, category/quartile contingency tests and Cramér’s V, Kruskal–Wallis group comparisons, missingness associations, and within-stratum follow-ups. Profiling includes missingness, duplicates, constants, quartiles/IQR outliers, and category imbalance.

## Reports and reproducibility

Download a bundle with Markdown report, PNGs, structured evidence, source, safe replay scripts, Jupyter notebook, and all generated execution attempts, including failures. Generated-source files are labeled untrusted; replay scripts/notebooks call `DockerSandbox`. To replay, install Pico, build the image, and place the original CSV beside the exported notebook as `dataset.csv`.

The original CSV is excluded. Evidence tables, correlation points, logs, generated source, and charts can still include source values; treat exports as sensitive as the dataset. Generated executions record the resolved worker image ID. Model choices depend on the local model; catalog demo runs are deterministic.

All results are exploratory associations. Adaptive selection, multiple comparisons, sparse cells, missingness, possible leakage, and model assumptions limit inference. P-values are raw. Execution and schema validation do not establish statistical correctness or causation; validate candidate findings independently.

## Development

```sh
pytest -q -m 'not docker'
ruff check src tests app.py sandbox/worker.py examples
ruff format --check src tests app.py sandbox/worker.py examples

# Requires running Docker and the built image; explicitly enabled tests fail if unavailable.
PICO_RUN_DOCKER_TESTS=1 pytest -q -m docker
```

A Makefile provides matching `run`, `test`, `lint`, `sandbox-image`, and `test-sandbox` targets. CI checks formatting and unit/UI tests on Python 3.11/3.14 and executes real-container tests in a dedicated Linux job.

The code separates domain records, model transport, planning policy, catalog methods, generated execution, protocol validation, UI/state, charts, and exports. See [architecture](docs/architecture.md) for responsibilities, resource limits, trust boundaries, and extension points.
