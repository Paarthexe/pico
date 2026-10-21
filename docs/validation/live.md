# Live model validation

Date: 2026-10-21. Ollama model: `qwen2.5-coder:7b`, digest `dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364`.

The real model produced three distinct Python investigations, executed in Docker in 179.37 seconds: income quartiles, purchase-frequency quartiles, and age quartiles against churn. Each used 2,235 observations. Independently recomputing chi-square tests from the returned counts reproduced all three p-values; each table also contained exactly 2,235 observations.

See [recorded results](live-followup.json) and the original, inert [generated source](generated/). This run used commit `16ac274`. The recording includes token counts, timings, immutable sandbox image identity, and the executed conclusions. It contains no recorded-response substitution.

The model incorrectly claimed an age association despite p=0.99009. A regression test and deterministic review guard now flag that interpretation as unsupported, preserve the original text, and replace the displayed conclusion. The JSON explicitly distinguishes the original run from its subsequent review with this guard.

Earlier trials exposed timeouts, invalid structured responses, nested result tables, and duplicate proposals. The provider now generates ordinary fenced source, distinguishes repair from successful follow-up, and feeds validation feedback into bounded proposal retries. Invalid output remains rejected. The default timeout is configurable and generation has a bounded token budget.

These are targeted smoke tests, not a general model benchmark. Code execution, scalar validation, and p-value correction cannot establish scientific correctness. The model can still produce misleading chart labels, omit relevant controls, or report inaccurate limitations. In one chart trial, it labeled counts as rates; visual inspection caught this, and the artifact must not be treated as a validated rate plot. Generated figures and interpretations require review.

Holm correction covers the reported reliable test family. Adaptive selection, unreported tests, confounding, timing leakage, independence, and missingness still require independent validation and domain knowledge. Leakage checks are heuristic; passing them does not prove absence of leakage.

Reproduce with `python scripts/validate_live.py --model qwen2.5-coder:7b`. The optional `--inject-first-error` deliberately prepends a labeled failure to the first execution; the saved original model source and executed injected source remain separate. Live model output may vary.
