"""Analytical acceptance cases with known ground truth and adversarial reports."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pico.data import profile_dataset
from pico.models import Evidence, ExecutionRecord, Hypothesis, Investigation
from pico.plotting import charts_for
from pico.review import review_investigation
from pico.verification import verify_contingency


def dataset(counts):
    return pd.DataFrame(
        [(str(i), str(j)) for i, row in enumerate(counts) for j, n in enumerate(row) for _ in range(n)],
        columns=["x", "y"],
    )


def reported(df, test="pearson_chi2", bins=0):
    paired = df[["x", "y"]].dropna().copy()
    if bins:
        paired["x"] = pd.qcut(paired.x, 4, labels=False)
    counts = pd.crosstab(paired.x.astype(str), paired.y.astype(str))
    result = (
        stats.chi2_contingency(counts, correction=False)
        if test == "pearson_chi2"
        else stats.fisher_exact(counts, alternative="two-sided")
    )
    return Evidence(
        "A01",
        asdict(Hypothesis("python", "x", "y", "Fixture")),
        len(paired),
        float(result.statistic) if np.isfinite(result.statistic) else None,
        float(result.pvalue),
        None,
        "An exploratory comparison was computed.",
        ["Fixture"],
        [dict(group=g, category=c, count=int(counts.loc[g, c])) for g in counts.index for c in counts.columns],
        "untrusted_source_must_never_run()",
        execution=ExecutionRecord("A01", "", "ok", 0),
        method=test,
        assumptions=["independence"],
        diagnostics=dict(verification_test=test, verification_missing="complete_case", verification_bins=bins),
    )


@pytest.mark.parametrize(
    "counts,test",
    [
        ([[20, 20], [20, 20]], "pearson_chi2"),  # exact null
        ([[80, 20], [20, 80]], "pearson_chi2"),  # strong signal
        ([[30, 20, 10], [10, 20, 30]], "pearson_chi2"),  # multiclass
        ([[1, 9], [8, 2]], "fisher_exact"),  # sparse
        ([[0, 10], [10, 0]], "fisher_exact"),  # infinite odds ratio
    ],
)
def test_recomputed_known_tables(counts, test):
    df = dataset(counts)
    e = reported(df, test)
    assert verify_contingency(df, e)["status"] == "matched"


def test_quantiles_and_pairwise_missingness():
    rng = np.random.default_rng(71)
    df = pd.DataFrame(dict(x=rng.normal(size=400), y=rng.choice(["a", "b"], 400)))
    df.loc[:10, "x"] = np.nan
    df.loc[20:29, "y"] = None
    e = reported(df, bins=4)
    check = verify_contingency(df, e)
    assert check["status"] == "matched" and check["n"] == 379


@pytest.mark.parametrize(
    "mutation", ["n", "p", "statistic", "count", "rate", "duplicate", "missing_cell", "missing_policy", "bins"]
)
def test_corrupted_reports_are_rejected(mutation):
    df = dataset([[80, 20], [20, 80]])
    e = reported(df)
    if mutation == "n":
        e.n -= 1
    elif mutation == "p":
        e.p_value = 0.9
    elif mutation == "statistic":
        e.statistic = 1
    elif mutation == "count":
        e.table[0]["count"] += 1
    elif mutation == "rate":
        e.table[0]["count"] = 0.8
    elif mutation == "duplicate":
        e.table.append(e.table[0])
    elif mutation == "missing_cell":
        e.table.pop()
    elif mutation == "missing_policy":
        e.diagnostics["verification_missing"] = "impute"
    else:
        e.diagnostics["verification_bins"] = 1000000
    check = verify_contingency(df, e)
    assert check["status"] == "failed"
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.status == "unsupported" and e.review.adjusted_p_value is None
    assert run.multiplicity["family_size"] == 0


def test_sparse_pearson_rejected_even_when_model_claims_dense_counts():
    df = dataset([[1, 9], [8, 2]])
    e = reported(df)
    e.diagnostics["min_expected_count"] = 100
    assert verify_contingency(df, e)["status"] == "failed"


def test_arbitrary_method_is_explicitly_unverified():
    df = dataset([[20, 20], [20, 20]])
    e = reported(df)
    e.diagnostics["verification_test"] = "eval"
    assert verify_contingency(df, e)["status"] == "unverified"


def test_matched_calculation_does_not_endorse_false_association_or_causality():
    df = dataset([[20, 20], [20, 20]])
    e = reported(df)
    e.conclusion = "The predictor is associated with the outcome and causes it."
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.verification["status"] == "matched"
    assert e.review.status == "unsupported"
    assert {"association_claim", "causal_claim"} <= {i["code"] for i in e.review.issues}
    review_investigation(df, run)
    assert e.review.status == "unsupported"


def test_checked_chart_uses_host_counts_not_generated_artifacts():
    df = dataset([[80, 20], [20, 80]])
    e = reported(df)
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    charts = charts_for(e)
    assert charts[0][0] == "A01-checked-proportions.png"
    assert charts[0][1].startswith(b"\x89PNG")
