# Analytical verification evaluation — 2026-10-22

This is a targeted acceptance corpus, not a general analytical accuracy benchmark.

## What changed

Trusted host code rebuilds declared Pearson chi-square (no continuity correction) and two-sided 2×2 Fisher exact tests from original data. It compares the entire long-form count table, complete-case n, statistic and p-value. Pearson expected counts are recomputed rather than accepted from model diagnostics. Unsupported transformations or tests remain unverified; mismatches and unverified generated tests are excluded from Holm inference. The result contract does not validate model-reported effect sizes.

For matched calculations, the displayed finding is a host-generated numerical summary and chart of within-group proportions. Model conclusions and images are retained for audit. Successful arithmetic does not validate sampling, independence, confounding, missingness, causality, or arbitrary generated source.

## Live model results

Model: Ollama `qwen2.5-coder:7b`; immutable model and sandbox image digests, usage, durations, raw conclusions, and execution outcomes are recorded in each summary. Source snapshots are inert `.py.txt` files. These were real generations executed by Docker, not substituted responses. Working-tree changes were in progress during this evaluation; summaries' HEAD SHA is not a guarantee that every executed file matched that commit. Code was subsequently committed in `b72888d` and `28853c0`.

| Dataset | Rebuilt result | Original live acceptance |
| --- | --- | --- |
| Exact null | n=160, chi-square=0, p=1 | Failed due to a reviewer negation bug; fixed and rerun |
| Strong association | n=200, chi-square=72, p≈2.152e-17 | Passed after one failed execution |
| Sparse 2×2 | n=20, Fisher test | Passed; small-sample warnings retained |
| Missing predictor | n=190 of 200, chi-square≈90.168, p≈2.188e-21 | Passed after one failed execution; missingness warning retained |

The [original evaluation](original-evaluation.json) preserves the null failure. The reviewer incorrectly treated “Insufficient evidence to claim a significant association” as a positive claim. The fix handles uncertainty and clause boundaries, with regression tests. [Posthoc review](posthoc-review.json) is explicitly separate from the original run; it accepts all four calculations with cautions. A [fresh live null rerun](fresh-null/summary.json) passes with the fixed reviewer and host-generated summary. Earlier model failures included gratuitous qcut binning of a binary predictor; these remain in source/summary records.

The model also used generic insufficient-evidence wording on strong signals. We do not count that prose as verified: the main finding now reports independently computed numbers and retains the original interpretation separately. Regex interpretation checks remain heuristics and can have false positives/negatives.

## Regression coverage

89 local tests pass, including 11 real Docker tests. New cases cover null and strong signals, multiclass counts, Fisher with infinite odds ratio, pairwise missingness, quantiles, fabricated n/p/statistics/counts, rates passed as counts, duplicate/missing cells, unsupported transforms, secondary unverified p-values, normalized chart bars/labels, target recodings, selection warnings, Simpson reversal, negated claims, and preservation of original prose across repeated review.

Independent verification currently covers contingency tests only. Expand the fixed recipes and evaluation corpus before relying on other generated methods. These deterministic synthetic cases do not measure accuracy on diverse real-world datasets.

## Reproduce

Prepare fixtures: `.venv/bin/python scripts/evaluate_analytical.py`.

Run live evaluation: `.venv/bin/python scripts/evaluate_analytical.py --live --output /tmp/pico-analytical-evaluation`.

The default budget is two proposals per dataset; one accepted calculation is required. A failed case makes the command exit unsuccessfully; its logs and evidence remain available. Do not treat a later posthoc review as a fresh generation.

Statistical definitions follow [SciPy Pearson chi-square](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html) and [SciPy Fisher exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).
