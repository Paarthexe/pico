# Pico

**Pattern Investigation and Computational Observation agent**

Pico investigates CSV data through a repeated analysis process:

```text
Inspect data → Select a hypothesis → Execute an analysis → Review evidence → Select the next analysis
```

You can use built-in statistical tools without a model.
You can also use a local model to write Python analyses.
Pico executes that Python in restricted Docker containers.

Each finding contains measured values, an evidence table, source code, and limitations.
The report also retains execution failures and review results.

**Pico supports exploratory analysis. It does not establish causation or guarantee a correct scientific interpretation.**

## Contents

- [Capabilities](#capabilities)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Enable generated Python](#enable-generated-python)
- [Investigate your data](#investigate-your-data)
- [Declare the study context](#declare-the-study-context)
- [Understand statistical verification](#understand-statistical-verification)
- [Reports and replay](#reports-and-replay)
- [Python interfaces](#python-interfaces)
- [Configuration](#configuration)
- [Container isolation and data handling](#container-isolation-and-data-handling)
- [Tests and evaluation](#tests-and-evaluation)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Known limits](#known-limits)
- [Technical terms](#technical-terms)

## Capabilities

| Capability | Current behavior |
| --- | --- |
| CSV inspection | Reports missing values, duplicate rows, constant columns, numeric summaries, and category counts. |
| Built-in analysis | Tests associations through a fixed statistical catalog. |
| Local model planning | Uses Ollama to select a hypothesis or generate an analysis. |
| Generated Python | Executes source in a new restricted Docker container for each attempt. |
| Error feedback | Returns execution errors and review issues to the model. |
| Independent verification | Recomputes supported statistical recipes from the original data. |
| Study context | Uses declared subjects, timestamps, and post-outcome columns to restrict analysis. |
| Uncertainty | Computes effect estimates and pointwise 95% intervals for supported numeric recipes. |
| Export | Packages reports, charts, evidence, source, execution records, and replay files. |

### Analysis engines

| Engine | Planner | Docker required | Ollama required |
| --- | --- | --- | --- |
| Catalog tools | Local adaptive rules | No | No |
| Catalog tools | Ollama local model | No | Yes |
| Generated Python (Docker) | Ollama local model | Yes | Yes |

Local rules use the selected outcome.
They do not interpret arbitrary questions.
Use Ollama when the question must affect hypothesis selection.

The generated engine supports analyses outside the catalog.
Independent verification covers only the recipes listed below.

## Requirements

| Component | Requirement |
| --- | --- |
| Host Python | Python 3.11 or newer. CI tests Python 3.11 and 3.14. |
| Docker | A local Linux container runtime with the required isolation and resource controls. |
| Worker image | Build `pico-sandbox:1` before generated execution. |
| Ollama | A local service at `http://127.0.0.1:11434`, with an installed model. |
| Browser tests | Node.js, npm, and Playwright Chromium. CI uses Node.js 22. |

Docker and Ollama are optional for the catalog demo.
The commands below use a macOS or Linux shell.

## Quick start

### 1. Get the project

```sh
git clone https://github.com/Paarthexe/pico.git
cd pico
```

Use an existing checkout if you already have the project.
Run the remaining commands from the project root.

### 2. Create a Python environment

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the application

```sh
python -m pip install -e '.[dev]'
```

The `dev` extra installs the test and formatting tools.

### 4. Start the application

```sh
python -m streamlit run app.py
```

Open [the local application](http://localhost:8501).
The synthetic customer dataset is enabled initially.

### 5. Run the demo

1. Keep **Use synthetic customer demo** enabled.
2. Select **Catalog tools**.
3. Select **Local adaptive rules**.
4. Set the analysis budget to five.
5. Select `churn` as the outcome.
6. Select **Start investigation**.

The demo contains 2,400 synthetic customer records.
It includes age, income, purchase frequency, churn, and customer segment.

Purchase frequency affects churn in the data generator.
Income is associated with purchase frequency.
The demo can therefore show an income association that changes within purchase-frequency strata.
This comparison is descriptive, not a causal estimate.

## Enable generated Python

### 1. Start Docker

Start your Docker runtime.
Make sure that its Linux daemon is available.

```sh
docker info
```

Pico checks the built-in seccomp profile and resource-limit support before model generation.
An unsupported runtime stops the generated engine.

### 2. Build the worker image

```sh
docker build -f sandbox/Dockerfile -t pico-sandbox:1 .
```

The worker uses Python 3.12.
Its scientific packages include pandas, NumPy, SciPy, Matplotlib, Seaborn, statsmodels, and scikit-learn.

Rebuild the image after changes to worker files or worker dependencies.
Pico does not pull an image during an investigation.

### 3. Start Ollama

Install [Ollama](https://docs.ollama.com/).
Start its local service.
If no service is running, use a separate terminal:

```sh
ollama serve
```

Use the existing service if Ollama already runs in the background.

### 4. Install a model

```sh
ollama pull qwen2.5-coder:7b
ollama list
```

`qwen2.5-coder:7b` is the default model name.
Recorded live checks used this model.
Those checks do not establish general model reliability.
Memory use and response time depend on the model and the computer.

### 5. Select the generated engine

1. Select **Generated Python (Docker)**.
2. Enter the installed model name in **Ollama model**.
3. Select an outcome column.
4. Enter a specific analysis question.
5. Declare the study context, if known.
6. Select **Start investigation**.

Example question:

> Compare income quartiles with churn. Use the result to select a follow-up involving purchase frequency. Report associations only.

### Attempt budget and repair

The budget permits one to five proposals.
A proposal can fail before execution.
It can also fail during execution or statistical review.

The model receives relevant failures and review issues before its next proposal.
A retry consumes the remaining budget.
Successful execution does not guarantee an accepted interpretation.

Pico does not execute identical successful source twice.
Different source can still repeat the same statistical comparison.
Repeated reported tests can remain in the adjustment family.

### macOS temporary-directory access

The Docker runtime must access the canonical host temporary directory.
Find that directory with this command:

```sh
python -c 'import pathlib, tempfile; print(pathlib.Path(tempfile.gettempdir()).resolve())'
```

For Colima, share that directory read-only when you configure the VM.
A remote Docker daemon cannot use these input files without access to the same host paths.
See [data and resource handling](docs/architecture.md#data-and-resource-handling).

## Investigate your data

### Accepted input

| Property | Limit or behavior |
| --- | --- |
| File type | UTF-8 CSV |
| File size | Up to 20 MiB |
| Row count | 1 to 100,000 |
| Column count | Up to 100 |
| Outcome | Must contain at least two distinct nonmissing values |
| Infinite numeric values | The CSV loader replaces them with missing values. |

Pico uses pandas to infer CSV column types.
Check the displayed sample and schema before analysis.
Numeric-looking category codes can become numeric columns.

### Procedure

1. Upload the CSV in the sidebar.
2. Disable the synthetic demo if it remains enabled.
3. Open **Inspect dataset and column health**.
4. Check column types, missing values, and category counts.
5. Select **Outcome to explore**.
6. Enter a question, if applicable.
7. Select the engine and budget.
8. Complete **Study context**, if applicable.
9. Select **Start investigation**.
10. Inspect the findings, review notes, and source.
11. Download the investigation bundle when required.

The question alone does not change the selected outcome.
Check both fields before you start.

Changed settings mark existing results as stale.
Start a new investigation to apply the changes.
A dataset change clears the previous results and study controls.

## Declare the study context

Open **Study context** before you start an investigation.
Leave a field unspecified when you do not know its meaning.

| Field | Purpose | Analysis restriction |
| --- | --- | --- |
| Subject identifier | Identifies the person, account, device, or other repeated unit. | Excluded from eligible predictors and follow-up controls. |
| Observation timestamp | Identifies when the observations were recorded. | Excluded from eligible predictors and follow-up controls. |
| Outcome timestamp | Identifies when the outcome occurred. | Excluded from eligible predictors and follow-up controls. |
| Columns recorded after the outcome | Identifies fields unavailable before the outcome. | Excluded from predictors and withheld from generated execution. |

Use different columns for the subject and timestamp roles.
Do not assign the outcome itself to a context role.
Use explicit date strings for timestamps, such as `2026-10-22T09:00:00Z`.
Pico does not guess the units of numeric timestamps.

### Inference restrictions

The review suppresses independent-row inference when it detects:

- Repeated subject identifiers.
- Missing subject identifiers.
- Missing, numeric, or unparseable timestamps.
- Observation timestamps at or after the outcome timestamp.

A numerical recipe can still match under these conditions.
The estimate can remain descriptive.
Pico excludes its p-value from inferential adjustment and suppresses interval display.
The structured evidence retains computed nominal bounds for audit, with `inference_eligible=false`.

Valid chronology does not prove independence or predictor availability.
The application retains a temporal-design warning.
It does not fit repeated-measures or time-series models automatically.

Reports retain the context declarations.
Generated replay files preserve the post-outcome column exclusions.
Subject and timestamp columns remain available in the execution snapshot for audit.

## Understand statistical verification

### Catalog methods

The trusted catalog includes:

- Spearman correlation.
- Contingency tests for categories and numeric groups.
- Cramér's V for contingency associations.
- Kruskal–Wallis comparisons of numeric outcomes across groups.
- Missingness associations.
- Comparisons within control-variable strata.

Spearman samples of at most 500 rows use 1,999 reproducible pairing permutations.
Larger samples use an asymptotic p-value.
Sparse 2×2 tables use Fisher's exact test.
Larger sparse tables retain descriptive results without a chi-square p-value.
Kruskal–Wallis comparisons require at least five complete observations per group.

Catalog calculations run in the application process.
Model-generated Python runs only in Docker.

### Independently verified generated methods

The host rebuilds a declared recipe from the original data.
It does not execute the generated source to perform verification.

| Recipe | Required input | Host verification | Effect and interval |
| --- | --- | --- | --- |
| `pearson_chi2` | Two categorical variables; optional four quantile bins for numeric x | Counts, complete-case sample size, statistic, p-value, expected counts | Model effects and effect intervals are not verified. |
| `fisher_exact` | A 2×2 contingency table | Counts, complete-case sample size, two-sided p-value, statistic | Model effects and effect intervals are not verified. |
| `pearson_r` | Two numeric variables | Numeric summary, sample size, correlation, p-value | Pearson r and a Fisher-z 95% interval |
| `welch_t` | Two groups in x; numeric y | Group summaries, sample size, Welch statistic, p-value | Mean difference and a Welch 95% interval |
| `ols_simple` | Numeric x and y | Numeric summary, sample size, slope, p-value | Slope and a classical t-based 95% interval |

All recipes use pairwise complete cases.
Numeric recipes do not permit imputation, extra filtering, or binning.
Contingency verification accepts original categories or four quantile bins of x.
Each contingency dimension must contain 2 to 20 observed categories.
Pearson chi-square verification uses `correction=False` and requires expected counts of at least five.
Fisher verification uses `alternative='two-sided'` and requires a 2×2 table.

Welch effects use the second group mean minus the first group mean.
The host sorts group labels as strings to determine that order.

Simple OLS fits one predictor and an intercept.
It does not adjust for other variables.
It does not provide logistic, multivariable, or cluster-robust regression verification.

### Review states

Verification and interpretation have different review states.

| State | Meaning |
| --- | --- |
| Verification: `matched` | The reported calculation agrees with the host recipe. |
| Verification: `failed` | A supported recipe has invalid input or a numerical mismatch. |
| Verification: `unverified` | No supported verification recipe was supplied. |
| Interpretation: `caution` | No blocking review issue was detected. Assumptions still require review. |
| Interpretation: `unsupported` | A blocking issue prevents endorsement of the interpretation. |

A matched result does not establish a valid study design.
Unsupported generated methods can execute, but their tests do not enter inferential adjustment.

For matched results, Pico displays a host-generated numerical summary and chart.
The original model interpretation remains available separately.
Failed interpretations display a review notice.
The report retains their original text and numerical evidence.

### Effects and confidence intervals

For numeric recipes, the host computes effects and pointwise 95% intervals.
It also checks optional model-reported effects and interval bounds.
Partial or incorrect reported bounds fail verification.

The intervals depend on each method's assumptions.
These include independence, distributional assumptions, and a suitable model form.
Classical OLS intervals also assume constant error variance.

**These intervals are not adjusted for multiple testing or adaptive selection.**
An arithmetic match does not prove that the assumptions hold.
Study or test failures can suppress the inferential interval.

### Multiple testing and claim review

Pico applies Holm adjustment to eligible tests across the investigation.
The family includes trusted catalog tests and independently matched generated primary tests.
Failed assumptions or study checks can exclude a test.
Unverified generated secondary p-values do not enter the family.

The report preserves raw p-values, adjusted p-values, and family size.
Repeated reported tests can count separately.
Holm adjustment does not remove adaptive selection bias or validate unreported comparisons.

The reviewer also checks identifiers, target copies, target recodings, possible leakage, and missingness.
It checks reported convergence issues and some unsupported statistical claims.
These checks use rules and heuristics.
They can miss an error or flag a correct statement.

## Reports and replay

Select the report tab to download an investigation bundle.

```text
investigation.zip
├── README.md
├── report.md
├── evidence.json
├── investigation.ipynb
├── charts/
├── code/
├── replay/
└── attempts/
```

| Item | Contents |
| --- | --- |
| `report.md` | Findings, statistical review, context declarations, charts, and investigation log |
| `evidence.json` | Structured investigation records, verification details, and execution information |
| `investigation.ipynb` | Notebook with the appropriate replay interface |
| `charts/` | Exported charts, including trusted charts for matched recipes |
| `code/` | Original analysis source; generated files are marked untrusted |
| `replay/` | Replay wrappers for the selected execution engine |
| `attempts/` | Generated execution records, including failures |

### Replay a generated investigation

1. Install Pico in a Python environment.
2. Build the worker image.
3. Extract the investigation bundle.
4. Put the original CSV beside the notebook as `dataset.csv`.
5. Run a replay wrapper or a notebook code cell.

**Do not execute generated files under `code/` directly on the host.**
Generated replay wrappers send source to `DockerSandbox`.
They also remove the declared post-outcome columns.
A replay can return different results after a dependency or image change.
The evidence records the original worker image identity.

The bundle excludes the original CSV.
Evidence, logs, charts, and source can still contain source values.
Handle the bundle with the same care as the dataset.

## Python interfaces

### Run the catalog

```python
from pico.agent import investigate
from pico.data import demo_dataset
from pico.report import export_bundle

frame = demo_dataset()
run = investigate(frame, target="churn", budget=5)
bundle = export_bundle(run)
```

### Supply study context

```python
from pico.agent import investigate
from pico.data import load_csv
from pico.models import StudyContext

with open("dataset.csv", "rb") as source:
    frame = load_csv(source.read())

context = StudyContext(
    subject_id="customer_id",
    observation_time="observed_at",
    outcome_time="outcome_at",
    post_outcome_columns=("cancellation_reason",),
)

run = investigate(frame, target="churn", study_context=context)
```

The example requires those columns in your dataset.
Use `engine="python"` and `model="qwen2.5-coder:7b"` to select generated analysis.

### Execute source without a model

```python
from pathlib import Path

from pico.data import demo_dataset
from pico.sandbox import DockerSandbox

frame = demo_dataset()
source = Path("examples/adjusted_churn.py").read_text()
execution = DockerSandbox().execute(
    source,
    frame.to_csv(index=False).encode(),
    len(frame),
)
print(execution.status, execution.result, execution.error)
```

The example fits adjusted logistic regression inside Docker.
Its successful execution does not verify the regression method.
The direct executor does not apply `StudyContext` exclusions automatically.
Use `investigate()` when you require those restrictions and statistical review.

### Worker result contract

Generated source can use `load_dataset()` and `save_result()` from `pico_tools` inside the container.
These helpers are container-only utilities.

A result requires:

- `n`: a positive integer no greater than the dataset row count.
- `conclusion`: nonempty text.
- `limitations`: one to twenty text entries.
- `table`: one to one thousand aggregate rows with scalar values.

Optional fields include `statistic`, `p_value`, `effect`, `method`, `assumptions`, and `diagnostics`.
Numeric fields must be finite or null.
A p-value must be between zero and one.
stdout and stderr are diagnostics, not substitutes for the result.

Use the declared verification recipe to obtain independent verification.
See the [planner contract](src/pico/planning.py) and [protocol validator](src/pico/sandbox/protocol.py) for exact fields and limits.

## Configuration

| Setting | Default | Effect |
| --- | --- | --- |
| `PICO_OLLAMA_MODEL` | `qwen2.5-coder:7b` | Initial model name in the UI |
| `PICO_OLLAMA_TIMEOUT_SECONDS` | `180` | Model response timeout; allowed range: 5 to 600 seconds |
| `PICO_RUN_DOCKER_TESTS` | Unset | Set to `1` to enable actual Docker tests |
| `PICO_PYTHON` | `python3` | Python executable for the browser test server |
| `PICO_BROWSER_PORT` | `8501` | Application port for browser tests |
| `GH_TOKEN` or `GITHUB_TOKEN` | Unset | Optional authentication for the CI status script |

Set application values before you start Streamlit:

```sh
export PICO_OLLAMA_MODEL=qwen2.5-coder:7b
export PICO_OLLAMA_TIMEOUT_SECONDS=180
python -m streamlit run app.py
```

[`.env.example`](.env.example) lists the model settings.
The application does not automatically load a `.env` file.
Model HTTP calls use the local Ollama endpoint.
The current application does not expose a remote provider setting.

## Container isolation and data handling

### Isolation controls

Each generated attempt uses a new container with:

- No network access.
- A non-root user.
- Read-only input files and root filesystem.
- No Linux capabilities or new privileges.
- The built-in Docker seccomp profile.
- Limited writable temporary filesystems.
- CPU, memory, process, file, and output limits.

The container receives the dataset snapshot and source.
Pico does not forward credentials, the Docker socket, or the general host environment.
The application removes the container and its descendants after execution.
It reports cleanup failures instead of reporting success.

Missing Docker, a missing image, or unsupported isolation stops generated execution.
There is no host-execution fallback.
The host and Docker runtime remain trusted infrastructure.
Container isolation does not prove protection against kernel vulnerabilities.

### Default execution limits

| Resource | Default |
| --- | --- |
| Wall time | 45 seconds per execution |
| CPU time | 30 seconds per process |
| CPU allocation | One core |
| Memory | 768 MiB; no additional swap allowance |
| Process limit | 64 |
| Writable output filesystem | 32 MiB |
| Temporary filesystem | 64 MiB |
| Per-file size | 8 MiB |
| Open file descriptors | 128 |

### Output limits

| Output | Limit |
| --- | --- |
| Generated source | 64 KiB |
| Result JSON | 256 KiB |
| stdout and stderr | 64 KiB each |
| PNG files | Four |
| Individual PNG | 2 MiB and 8 million pixels |
| Combined PNG data | 4 MiB |
| Transport response | 7 MiB |

[`SandboxLimits`](src/pico/config.py) permits bounded API overrides for selected execution resources.
These overrides are not UI settings.

### Data flow

The model receives schema, aggregate summaries, prior findings, source, and relevant error text.
Pico does not deliberately put raw CSV rows into prompts.
Category summaries, generated results, or errors can still contain dataset values.

Ollama receives requests through its local [chat API](https://docs.ollama.com/api/chat).
Docker receives a CSV snapshot after declared post-outcome exclusions.
The original dataframe remains available to trusted host verification.

Pico briefly writes execution inputs to host temporary files.
Normal cleanup removes those inputs after execution.
Abrupt application termination can prevent cleanup.
Container resource limits remain separate from application cleanup.

## Tests and evaluation

### Run local checks

```sh
python -m pytest -q -m 'not docker'
python -m ruff check src tests scripts app.py sandbox examples
python -m ruff format --check src tests scripts app.py sandbox examples
```

### Run actual Docker tests

Start Docker and build the image first.

```sh
PICO_RUN_DOCKER_TESTS=1 python -m pytest -q -m docker
```

Run the complete Python suite with:

```sh
PICO_RUN_DOCKER_TESTS=1 python -m pytest -q
```

Explicitly enabled Docker tests fail when the runtime or image is unavailable.
They do not substitute a host executor.

### Run browser checks

```sh
npm ci
npx playwright install chromium
PICO_PYTHON=.venv/bin/python npm run test:browser
```

Use a separate port to test a fresh server:

```sh
PICO_PYTHON=.venv/bin/python PICO_BROWSER_PORT=8502 CI=1 npm run test:browser
```

The fresh-server command requires an unused port.
CI installs Chromium system dependencies with `playwright install --with-deps chromium`.

Browser checks cover desktop and mobile layouts, study controls, navigation, chart descriptions, report download, and axe accessibility scans.
They do not establish full accessibility conformance.

### Run a live model check

Start Ollama and Docker before this command:

```sh
python scripts/validate_live.py \
  --model qwen2.5-coder:7b \
  --budget 3 \
  --output /tmp/pico-live-validation
```

The script uses actual model responses and actual Docker execution.
It saves source, execution records, model identity, usage, findings, and an export bundle.
By default, it requires at least two executed findings.

Use `--dataset` and `--target` for another CSV.
Use `--question` to supply a specific question.
Use `--minimum-findings` to change the acceptance count.
Use `--require-verified` to require matched findings without blocking review issues.
Use `--inject-first-error` to test a deliberately inserted failure.
The script records that injection explicitly.

The live validation CLI reads a supplied CSV directly with pandas.
It does not use the UI upload validation or provide study-context options.
Use the Python investigation interface for context-aware validation.

### Run the synthetic evaluation

Prepare the null, signal, sparse, and missing-data fixtures:

```sh
python scripts/evaluate_analytical.py
```

Run actual model investigations on those fixtures:

```sh
python scripts/evaluate_analytical.py \
  --live \
  --model qwen2.5-coder:7b \
  --output /tmp/pico-analytical-evaluation
```

Each case requires an independently matched finding without blocking review issues.
The default budget is two proposals per case.
The command retains failed results and exits unsuccessfully if a case fails.
This small synthetic suite is not a general model benchmark.

### Check GitHub Actions

```sh
python scripts/check_ci.py
```

The script uses a supplied token or existing Git authentication.
It reads workflow and job metadata for the current commit.
It does not print authentication values.

CI has four jobs:

1. Python 3.11 unit and UI checks.
2. Python 3.14 unit and UI checks.
3. Actual Docker execution and isolation checks.
4. Chromium browser and accessibility checks.

### Recorded validation

The implementation at `a926ea2` passed 129 Python tests and all CI jobs.
The browser suite contained seven tests.
Counts describe that revision, not future changes.

- [Browser validation](docs/validation/browser.md)
- [Initial live-model and repair records](docs/validation/live.md)
- [Analytical verification cases](docs/validation/analytical-2026-10-22/README.md)
- [Numeric methods and study context](docs/validation/numeric-study-2026-10-22/README.md)

## Architecture

```text
app.py
└── ui/                         Controls, rendering, and session state
    └── agent.py                Investigation orchestration
        ├── planning.py         Candidate selection and proposal validation
        ├── analysis.py         Trusted catalog calculations
        └── python_agent.py     Generated execution and repair loop
            ├── ollama.py       Local model transport
            └── sandbox/        Docker lifecycle and output validation

review.py                       Claim checks and investigation-wide adjustment
verification.py                 Independent contingency verification and dispatch
verification_numeric.py         Independent numeric verification
study.py                        Subject and chronology checks
models.py                       Shared domain records
plotting.py                     Catalog, generated, and trusted verification charts
report.py                       Report, bundle, notebook, and replay generation
```

The container entry point is [`sandbox/worker.py`](sandbox/worker.py).
Its helper module is [`sandbox/pico_tools.py`](sandbox/pico_tools.py).
Neither file executes generated Python in the application process.

See [the architecture document](docs/architecture.md) for lifecycle details and extension points.
Project-specific design decisions are stored in [`.tastemaker/`](.tastemaker/).
That directory does not replace the application specification or test results.

### Make targets

| Target | Action |
| --- | --- |
| `make install` | Install the application and development dependencies |
| `make run` | Start Streamlit |
| `make test` | Run tests without Docker |
| `make lint` | Check code style and formatting |
| `make sandbox-image` | Build the worker image |
| `make test-sandbox` | Run actual Docker tests |

Set `PYTHON` to select an interpreter, for example `make test PYTHON=.venv/bin/python`.

## Troubleshooting

| Symptom | Check or action |
| --- | --- |
| `ModuleNotFoundError: pico` | Activate the environment and run `python -m pip install -e '.[dev]'`. |
| Docker is unavailable | Start the local runtime and inspect `docker info`. |
| The worker image is missing | Build `pico-sandbox:1` from the project root. |
| Docker cannot mount an input file | Check canonical temporary-directory sharing, especially with Colima. |
| Ollama connection fails | Check the service at `127.0.0.1:11434`. |
| The model is unavailable | Inspect `ollama list` and enter an installed model name. |
| A model request times out | Check model load and latency; use a permitted timeout setting if required. |
| Generated output fails validation | Inspect the failed attempt, source, error, and result contract. |
| A result is unverified | Check whether its recipe and transformations have independent verification support. |
| A p-value or interval is excluded | Inspect study checks, verification results, and statistical review notes. |
| A timestamp check fails | Use explicit dates and check missing values, units, time zones, and event order. |
| Old results remain visible | Start a new investigation after changing the settings. |
| A browser test cannot start | Check the Python executable, Chromium installation, and application port. |

Inspect the execution trail before changing a limit.
A larger timeout does not correct invalid code or an unsuitable statistical method.

## Known limits

- The model can choose an unsuitable test or return misleading prose.
- Interpretation checks use heuristics and can produce false positives or false negatives.
- Independent verification covers fixed recipes, not arbitrary generated methods.
- Matched arithmetic does not verify sampling, independence, confounding, or measurement quality.
- Pointwise intervals do not account for model-driven selection or multiple comparisons.
- Study declarations depend on information supplied by the user.
- Undeclared leakage and temporal dependence can remain undetected.
- Repeated-measures, cluster-aware, time-series, logistic, and multivariable regression verification are not implemented.
- The synthetic evaluation does not establish reliability across real-world datasets or repeated model runs.
- Authentication, multi-user admission controls, job cancellation, and production data-retention controls are outside the current application.

## Technical terms

| Term | Meaning in this project |
| --- | --- |
| Model | The local language model used through Ollama. |
| Catalog | The application's fixed collection of trusted statistical methods. |
| Worker | The process that executes generated source inside Docker. |
| Recipe | A fixed statistical calculation that the host can rebuild independently. |
| Complete cases | Rows with nonmissing values in both analysis columns. |
| Effect | A numerical association estimate, such as a correlation or mean difference. |
| Pointwise interval | An interval for one estimate, without simultaneous coverage across the investigation. |
| Leakage | Use of information that would be unavailable at the intended analysis or prediction time. |
| Study context | User declarations about subjects, timestamps, and post-outcome fields. |
| Stale result | A result produced with settings different from the current settings. |
| Host | The trusted process and computer running the Pico application. |

This document uses short instructions and consistent technical terms in an ASD-STE100-inspired style.
It has not received a formal ASD-STE100 conformity assessment.
See [the ASD-STE100 overview](https://www.asd-ste100.org/about.html) for the controlled-language specification.
