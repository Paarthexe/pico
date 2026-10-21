from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pico.models import Evidence, ExecutionRecord, Hypothesis, Investigation, StudyContext
from pico.review import review_investigation
from pico.verification import verify_analysis


def numeric_fixture(recipe):
    rng = np.random.default_rng(71)
    a = rng.normal(size=120)
    b = 1.3 * a + rng.normal(size=120)
    df = pd.DataFrame(dict(x=a, y=b))
    if recipe == "welch_t":
        df.x = ["a"] * 60 + ["b"] * 60
        first, second = df.y.iloc[:60], df.y.iloc[60:]
        result = stats.ttest_ind(second, first, equal_var=False)
        statistic = result.statistic
        effect = second.mean() - first.mean()
        table = [
            dict(group=g, n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)))
            for g, v in [("a", first), ("b", second)]
        ]
    else:
        result = stats.pearsonr(a, b) if recipe == "pearson_r" else stats.linregress(a, b)
        statistic = result.statistic if recipe == "pearson_r" else result.slope
        effect = statistic
        table = [dict(n=len(df), x_mean=float(df.x.mean()), y_mean=float(df.y.mean()))]
    e = Evidence(
        "A01",
        asdict(Hypothesis("python", "x", "y", "fixture")),
        len(df),
        float(statistic),
        float(result.pvalue),
        float(effect),
        "An exploratory comparison was computed.",
        ["fixture"],
        table,
        "never_execute()",
        execution=ExecutionRecord("A01", "", "ok", 0),
        method=recipe,
        assumptions=["Independent observations"],
        diagnostics=dict(verification_test=recipe, verification_missing="complete_case", verification_bins=0),
    )
    return df, e


@pytest.mark.parametrize("recipe", ["pearson_r", "welch_t", "ols_simple"])
def test_numeric_recipes_effects_and_intervals_match_reference(recipe):
    df, e = numeric_fixture(recipe)
    check = verify_analysis(df, e)
    assert check["status"] == "matched"
    assert check["ci_low"] <= check["effect"] <= check["ci_high"]
    if recipe == "pearson_r":
        ci = stats.pearsonr(df.x, df.y).confidence_interval(0.95)
    elif recipe == "welch_t":
        ci = stats.ttest_ind(df.y.iloc[60:], df.y.iloc[:60], equal_var=False).confidence_interval(0.95)
    else:
        # Independent matrix calculation rather than linregress for the OLS oracle.
        design = np.column_stack([np.ones(len(df)), df.x])
        beta = np.linalg.lstsq(design, df.y, rcond=None)[0]
        residual = df.y - design @ beta
        se = np.sqrt((residual @ residual / (len(df) - 2)) * np.linalg.inv(design.T @ design)[1, 1])
        radius = stats.t.ppf(0.975, len(df) - 2) * se
        ci = (beta[1] - radius, beta[1] + radius)
    assert [check["ci_low"], check["ci_high"]] == pytest.approx(ci)
    e.diagnostics.update(ci_low=float(ci[0]), ci_high=float(ci[1]), confidence_level=0.95)
    assert verify_analysis(df, e)["status"] == "matched"


@pytest.mark.parametrize("recipe", ["pearson_r", "welch_t", "ols_simple"])
@pytest.mark.parametrize("field", ["p_value", "effect", "statistic", "n", "table", "ci", "partial_ci"])
def test_numeric_fabrications_fail(recipe, field):
    df, e = numeric_fixture(recipe)
    if field == "table":
        e.table[0]["n"] += 1
    elif field == "ci":
        e.diagnostics.update(ci_low=-999, ci_high=999, confidence_level=0.95)
    elif field == "partial_ci":
        e.diagnostics["ci_low"] = 0
    elif field == "n":
        e.n -= 1
    else:
        setattr(e, field, getattr(e, field) + 0.1)
    assert verify_analysis(df, e)["status"] == "failed"


@pytest.mark.parametrize("recipe", ["pearson_r", "welch_t", "ols_simple"])
def test_nonfinite_constant_and_missing_policy_fail_closed(recipe):
    df, e = numeric_fixture(recipe)
    df.loc[0, "y"] = np.inf
    assert verify_analysis(df, e)["status"] == "failed"
    df, e = numeric_fixture(recipe)
    df.y = 1.0
    assert verify_analysis(df, e)["status"] == "failed"
    df, e = numeric_fixture(recipe)
    e.diagnostics["verification_missing"] = "impute"
    assert verify_analysis(df, e)["status"] == "failed"


def test_repeated_subjects_keep_descriptive_effect_but_suppress_ci_and_holm():
    df, e = numeric_fixture("ols_simple")
    df["subject"] = np.repeat(np.arange(60), 2)
    run = Investigation({}, "y", "", "fixture", 1, evidence=[e], study_context=StudyContext(subject_id="subject"))
    review_investigation(df, run)
    assert e.review.verification["status"] == "matched"
    assert not e.review.verification["inference_eligible"]
    assert e.review.status == "unsupported" and e.review.adjusted_p_value is None
    assert "interval suppressed" in e.review.numerical_summary
    assert "95% CI [" not in e.review.numerical_summary
