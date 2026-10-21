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


def test_unverified_secondary_p_values_cannot_enter_correction():
    df = dataset([[20, 20], [20, 20]])
    e = reported(df)
    e.p_value = None
    e.table = [dict(term="invented", p_value=0.00001)]
    e.diagnostics = {}
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert run.multiplicity["family_size"] == 0
    assert e.review.verification["status"] == "unverified"


def test_checked_chart_axis_and_bar_values_are_proportions(monkeypatch):
    import matplotlib.pyplot as plt

    df = dataset([[80, 20], [20, 80]])
    e = reported(df)
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    captured = []
    original_close = plt.close

    def capture(fig=None):
        if hasattr(fig, "axes"):
            captured.extend(fig.axes)
        original_close(fig)

    monkeypatch.setattr(plt, "close", capture)
    charts_for(e)
    ax = captured[0]
    assert ax.get_ylim() == (0, 1)
    assert ax.get_ylabel() == "Proportion of y"
    assert sorted(p.get_height() for p in ax.patches) == pytest.approx([0.2, 0.2, 0.8, 0.8])


def test_recoded_target_and_differential_missingness_require_review():
    from pico.planning import candidates
    from pico.review import dataset_warnings

    df = dataset([[80, 20], [20, 80]])
    df["recoded"] = df.y.map({"0": "retained", "1": "lost"})
    df["selected"] = np.arange(len(df), dtype=float)
    df.loc[df.y == "1", "selected"] = np.nan
    warnings = dataset_warnings(df, "y")
    assert {"target_proxy", "differential_missingness"} <= {w["code"] for w in warnings}
    assert "recoded" not in {h.x for h in candidates(df, "y")}


def test_complete_case_match_still_warns_about_outcome_dependent_selection():
    df = dataset([[80, 20], [20, 80]])
    df.loc[df.index[(df.x == "0") & (df.y == "1")][:10], "x"] = None
    e = reported(df)
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.verification["status"] == "matched"
    assert any(i["code"] == "missingness" for i in e.review.issues)


def test_simpson_reversal_remains_unadjusted_even_with_matched_arithmetic():
    # In each stratum x=1 has lower event risk; aggregated x=1 has higher risk.
    low = dataset([[72, 8], [19, 1]]).assign(stratum="low")
    high = dataset([[2, 18], [16, 64]]).assign(stratum="high")
    df = pd.concat([low, high], ignore_index=True)
    aggregate = df.assign(event=df.y == "1").groupby("x").event.mean()
    assert aggregate["1"] > aggregate["0"]
    for _, group in df.groupby("stratum"):
        rates = group.assign(event=group.y == "1").groupby("x").event.mean()
        assert rates["1"] < rates["0"]
    e = reported(df)
    # A method string alone must not bypass the confounding guard.
    e.method = "Adjusted regression"
    e.conclusion = "The predictor explains the association between treatment and outcome."
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.verification["status"] == "matched"
    assert e.review.status == "unsupported"
    assert {"unadjusted_comparison", "confounding_claim"} <= {i["code"] for i in e.review.issues}


def test_calculation_check_does_not_silently_reset_prior_assumption_failure():
    df = dataset([[80, 20], [20, 80]])
    e = reported(df)
    e.diagnostics["converged"] = False
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.verification["status"] == "matched"
    assert e.review.status == "unsupported" and e.review.adjusted_p_value is None


@pytest.mark.parametrize(
    "conclusion",
    [
        "Insufficient evidence to claim a significant association between x and y.",
        "We cannot conclude that x and y are associated.",
        "This test does not establish no association.",
    ],
)
def test_null_uncertainty_is_not_misclassified_as_positive_or_absence_claim(conclusion):
    df = dataset([[40, 40], [40, 40]])
    e = reported(df)
    e.conclusion = conclusion
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.status == "caution"


def test_negation_in_one_clause_does_not_hide_assertion_after_but():
    df = dataset([[40, 40], [40, 40]])
    e = reported(df)
    e.conclusion = "There is insufficient evidence, but x and y are associated."
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.status == "unsupported"


def test_matched_finding_uses_trusted_numerical_summary_and_keeps_model_text():
    df = dataset([[80, 20], [20, 80]])
    e = reported(df)
    original = "Insufficient evidence to claim a significant association between x and y."
    e.conclusion = original
    run = Investigation(profile_dataset(df), "y", "", "fixture", 1, evidence=[e])
    review_investigation(df, run)
    assert e.review.status == "caution"
    assert e.review.original_conclusion == original
    assert e.conclusion == e.review.numerical_summary
    assert "raw p=2.152e-17" in e.conclusion
    assert "Holm-adjusted p=" in e.conclusion
    review_investigation(df, run)
    assert e.review.original_conclusion == original
