# Live model validation

Date: 2026-10-21. Ollama model: `qwen2.5-coder:7b`, digest `dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364`.

The real model produced three distinct Python investigations, executed in Docker in 179.37 seconds: income quartiles, purchase-frequency quartiles, and age quartiles against churn. Each used 2,235 observations. Independently recomputing chi-square tests from the returned counts reproduced all three p-values; each table also contained exactly 2,235 observations.

See [recorded results](live-followup.json) and the original, inert [generated source](generated/). This run used commit `16ac274`. The recording includes token counts, timings, immutable sandbox image identity, and the executed conclusions. It contains no recorded-response substitution.

The model incorrectly claimed an age association despite p=0.99009. A regression test and deterministic review guard now flag that interpretation as unsupported, preserve the original text, and replace the displayed conclusion. The JSON explicitly distinguishes the original run from its subsequent review with this guard.

Earlier trials exposed timeouts, invalid structured responses, nested result tables, and duplicate proposals. The provider now generates ordinary fenced source, distinguishes repair from successful follow-up, and feeds validation feedback into bounded proposal retries. Invalid output remains rejected. The default timeout is configurable and generation has a bounded token budget.

These are targeted smoke tests, not a general model benchmark. Code execution, scalar validation, and p-value correction cannot establish scientific correctness. The model can still produce misleading chart labels, omit relevant controls, or report inaccurate limitations. In one chart trial, it labeled counts as rates; visual inspection caught this, and the artifact must not be treated as a validated rate plot. Generated figures and interpretations require review.

Holm correction covers the reported reliable test family. Adaptive selection, unreported tests, confounding, timing leakage, independence, and missingness still require independent validation and domain knowledge. Leakage checks are heuristic; passing them does not prove absence of leakage.

Reproduce with `python scripts/validate_live.py --model qwen2.5-coder:7b`. The optional `--inject-first-error` deliberately prepends a labeled failure to the first execution; the saved original model source and executed injected source remain separate. Live model output may vary.

## Live repair and chart check

The [broader trial](repair-broad/summary.json) recovered from the injected first failure, then exhausted its five-attempt budget on nested output tables. It failed its two-finding acceptance criterion. This failure is retained rather than counted as a successful complete investigation.

After explicit single-comparison, flat-table, and plot-normalization instructions, the [focused trial](repair-focused/summary.json) passed its deliberately narrower acceptance criterion: two attempts, first an explicitly injected `PICO_VALIDATION_INJECTED_FAILURE`, then one successful real-model repair. It took 149 seconds on commit `f7a6582`. The model generated both responses live; Docker executed both attempts.

The repaired source computed chi-square on raw counts and plotted row-normalized proportions. Its [PNG](repair-focused/income_quartile_churn_proportions.png) was visually inspected: each stack sums to 1 and the axis correctly says proportion. Table counts sum to the reported 2,400 observations, and recomputation reproduces p=8.255335186361497e-20. The generated prose rounds that p-value to `0.0000`; the evidence preserves its nonzero numeric value. This smoke test demonstrates one repair, not universal repair reliability.

Command: `python scripts/validate_live.py --model qwen2.5-coder:7b --budget 2 --minimum-findings 1 --inject-first-error --question 'Perform exactly one comparison: income quartiles against churn, x=income y=churn. Plot within-quartile churn proportions between 0 and 1 as a PNG; run chi-square on raw counts. Return one flat counts table and one scalar p_value.' --output /tmp/pico-live-repair-focused`.

Local validation: 59 tests passed, including 11 actual Docker tests. Five isolated Chromium browser tests passed. GitHub Actions passed all four jobs for `f7a6582`: [unit Python 3.11/3.14, Docker sandbox, and browser](https://github.com/Paarthexe/pico/actions/runs/39647735963).
