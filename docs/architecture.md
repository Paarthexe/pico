# Pico architecture

`app.py` composes the Streamlit presentation layer. It delegates all domain work to `investigate()`.

```text
ui/ (controls, views, session state)
  └─ agent.py (validation, event trail, budget)
       ├─ catalog loop → planning.py → analysis.py
       └─ python_agent.py → PythonPlanner → ollama.py
                           └─ DockerSandbox → container worker
                                   └─ protocol validation → evidence
models.py holds domain records; ports.py describes injectable interfaces.
plotting.py handles catalog figures or returned PNGs; report.py packages both.
```

## Responsibilities and extension points

- `models.py`: hypotheses, proposals, evidence, executions, artifacts, and investigations. No Streamlit or model-provider dependencies.
- `ports.py`: structural interfaces for model clients, proposal planners, and Python executors. Tests inject narrow fakes rather than patching the orchestration internals.
- `config.py`: application budgets, resource configuration, and output limits.
- `data.py`: validated CSV ingestion, profiling, and deterministic demo data.
- `analysis.py`: trusted statistical method registry and common evidence finalization. Add a catalog method by implementing a focused summary function and registering it.
- `planning.py`: candidate ranking, follow-up policy, model context, and generated-proposal validation. Planning never executes code.
- `ollama.py`: local HTTP transport and bounded structured responses. Replace `ModelClient` to add another provider.
- `agent.py`: public investigation entry point, input validation, and catalog orchestration.
- `python_agent.py`: generated-code execution, failure feedback, duplicate-source protection, and saving only successful evidence. `ProposalPlanner` and `PythonExecutor` are injectable.
- `sandbox/docker.py`: container lifecycle and bounded subprocess capture. Docker commands are argument lists; there is no host shell or host Python execution of generated code.
- `sandbox/protocol.py`: validates untrusted result scalars, sample counts, log sizes, filenames, base64, and PNG content/dimensions before findings or exports can consume them.
- `sandbox/worker.py`: standalone container entry point, child process, bounded file reads, and output serialization. It intentionally does not depend on the installed host package. Protocol version and limits are exercised by integration tests.
- `ui/`: native Streamlit presentation, styles, and session state. A result is stale whenever dataset, question, target, budget, model, or engine changes.
- `plotting.py` and `report.py`: one chart interface and safe replay wrappers for both engines. Generated code is source text passed to Docker in exported notebooks, not executed by the notebook kernel.

## Execution lifecycle

1. Verify a running Linux Docker daemon with its built-in seccomp profile, enforced resource-limit support, and a locally built worker image. Resolve the image to an immutable ID.
2. Stage a dataset snapshot and generated source in a private temporary directory. Only the two input files are bind-mounted, read-only, into the container; the staging parent stays mode 0700 on the host. Canonical host paths support macOS VM shares.
3. Create a fresh named container with no network, non-root user, no capabilities, no-new-privileges, read-only root, bounded tmpfs directories, and CPU/memory/PID/file limits. No credentials, Docker socket, working directory, or general host environment are forwarded.
4. Start the worker. It launches arbitrary Python inside the container and collects result.json, PNGs, stdout, and stderr. Python import restrictions are not used as an isolation claim.
5. The host captures a bounded response under a wall deadline and validates it. Only a successful validated response becomes a finding. Failures, source, and diagnostics stay in the execution history and inform replanning.
6. Force-remove the container in a `finally` block, killing descendants. Temporary inputs are removed after container cleanup. Cleanup failures stop the run visibly rather than reporting success.

## Trust and limitations

Docker and its daemon, the built worker image, and the application host are trusted. Generated Python and every returned value are untrusted. Linux containers share their VM/host kernel; this is a restricted container boundary, not a proof against kernel exploits. For adversarial multi-user hosting, operate workers on dedicated infrastructure with an appropriately stronger isolation boundary and admission/concurrency controls.

The executor accepts a local image chosen by the application operator, resolves it to a digest for each executor, and never automatically pulls an image during analysis. The multi-platform base image digest and scientific direct dependencies are pinned in `sandbox/requirements.txt`; the recorded image ID identifies the exact built environment, including transitive dependencies. Updating/rebuilding the image is an explicit operator action.

A model can still write statistically inappropriate code or misstate results. Schema checks establish output shape, not scientific truth. Every finding keeps its code, numerical table, execution identity, and caveats for review. Raw p-values remain exploratory under adaptive selection.

## Data and resource handling

Inputs are briefly written to disposable host temporary files; outputs remain on container tmpfs and return as bounded bytes. There is no writable host output mount or archive extraction. Temporary input files are removed after execution, including errors and timeouts. As with any process, force-killing the application itself can leave its temporary files; container CPU/memory/PID limits continue to apply, but application-level cleanup cannot run after abrupt host termination.

The Docker daemon must be able to bind-mount the host's canonical temporary directory. Docker Desktop normally shares macOS temporary paths. For Colima, share the directory shown by `python -c 'import tempfile,pathlib; print(pathlib.Path(tempfile.gettempdir()).resolve())'` when starting the VM, read-only. Remote Docker daemons without access to those input paths are unsupported.

Execution defaults: 45 seconds wall time, 30 seconds CPU per process, one CPU core, 768 MB memory with no extra swap, 64 processes, 32 MB output tmpfs, 64 MB temporary tmpfs, 8 MB per file, 128 file descriptors, and no core dumps. Container setup/cleanup have separate bounded CLI timeouts. Model HTTP calls have a separate 45-second network timeout. Overrides can only stay within validated bounds.

Output defaults: 64 KB source, 256 KB result JSON, 64 KB each stdout/stderr, four PNG charts up to 2 MB each/4 MB total and 8 million pixels each, and a bounded 7 MB transport response. Nonregular output files, symlinks, invalid images, nonfinite numbers, impossible n, nested table cells, and malformed JSON cannot become evidence.

## Verification

Unit and Streamlit tests cover both engine selections, settings invalidation, provider errors, planner output, repair feedback, source deduplication, evidence provenance, safe export wrappers, and protocol rejection. Real Docker integration tests compute known values/charts and exercise no-network access, host/input/root restrictions, runtime errors, memory exhaustion, oversized files, symlink rejection, malformed results, and timeout/descendant cleanup. Explicitly enabling Docker tests fails if the runtime or image is unavailable; it never silently substitutes mocks or host execution.
