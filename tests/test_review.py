from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pico.analysis import execute_analysis
from pico.data import profile_dataset
from pico.models import Evidence, ExecutionRecord, Hypothesis, Investigation
from pico.planning import candidates
from pico.review import dataset_warnings, holm_adjust, review_investigation


def frame():
    return pd.DataFrame({"x": np.arange(80) * 2 + 1, "outcome": ["a", "b"] * 40})


def finding(index=1, p=0.01, conclusion="An exploratory association was measured.", table=None):
    return Evidence(
        f"A{index:02}",
        asdict(Hypothesis("python", "x", "outcome", "test")),
        80,
        1.0,
        p,
        0.2,
        conclusion,
        ["Exploratory fixture."],
        table or [{"estimate": 0.2}],
        'df["x"]',
        execution=ExecutionRecord(f"A{index:02}", 'df["x"]', "ok", 0.1),
        method="Fixture",
        assumptions=["Independent observations are unverified."],
    )


def run_with(*evidence):
    return Investigation(profile_dataset(frame()), "outcome", "", "test", 5, evidence=list(evidence))


def test_holm_known_values_order_empty_and_validation():
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert holm_adjust([0.9, 0.9]) == [1.0, 1.0]
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_correction_covers_primary_and_table_tests_and_exportable_review():
    first, second = finding(table=[{"term": "extra", "p_value": 0.03}]), finding(2, 0.04)
    run = run_with(first, second)
    review_investigation(frame(), run)
    assert run.multiplicity["family_size"] == 3
    assert first.review.adjusted_p_value == pytest.approx(0.03)
    assert second.review.adjusted_p_value == pytest.approx(0.06)
    assert first.review.adjusted_tests[1]["adjusted_p_value"] == pytest.approx(0.06)
    assert "Adaptive selection" in run.multiplicity["limitation"]


def test_later_tests_revise_unsupported_significance_without_losing_original():
    first = finding(p=0.04, conclusion="A statistically significant association was observed.")
    run = run_with(first)
    review_investigation(frame(), run)
    assert first.review.status == "caution"
    run.evidence.append(finding(2, 0.03))
    review_investigation(frame(), run)
    assert first.review.status == "unsupported"
    assert "statistically significant" in first.review.original_conclusion
    assert "requires review" in first.conclusion
    review_investigation(frame(), run)
    assert "statistically significant" in first.review.original_conclusion


def test_causal_claim_flag_and_negated_causality():
    unsafe = finding(conclusion="Income causes the outcome and proves the mechanism.")
    safe = finding(2, conclusion="This does not prove causality; the result is associational.")
    run = run_with(unsafe, safe)
    review_investigation(frame(), run)
    assert unsafe.review.status == "unsupported"
    assert safe.review.status == "caution"


def test_identifier_target_copy_and_possible_post_outcome_checks():
    df = frame().assign(customer_id=np.arange(80), copied_outcome=frame()["outcome"], post_outcome_score=1)
    warnings = dataset_warnings(df, "outcome")
    assert {"identifier", "target_copy", "post_outcome"} <= {w["code"] for w in warnings}
    assert not any(h.x in {"customer_id", "copied_outcome"} for h in candidates(df, "outcome"))
    e = finding()
    e.code = 'df[["customer_id", "x"]]'
    run = Investigation(profile_dataset(df), "outcome", "", "test", 1, evidence=[e])
    review_investigation(df, run)
    assert any(i["code"] == "identifier" for i in e.review.issues)
    assert e.review.status == "unsupported"


def test_nonconverged_model_p_values_are_not_used_for_inference():
    e = finding(conclusion="A statistically significant relationship was observed.")
    e.diagnostics = {"converged": False}
    run = run_with(e)
    review_investigation(frame(), run)
    assert e.review.adjusted_p_value is None
    assert not e.p_value_reliable
    assert run.multiplicity["family_size"] == 0
    assert e.review.status == "unsupported"


def test_sparse_two_by_two_uses_fisher_and_larger_sparse_table_suppresses_p():
    df = pd.DataFrame({"x": ["a"] * 9 + ["b"], "y": ["no"] * 8 + ["yes", "yes"]})
    e = execute_analysis(df, Hypothesis("groups", "x", "y", "test"))
    expected = stats.fisher_exact(pd.crosstab(df.x, df.y)).pvalue
    assert e.p_value == pytest.approx(expected)
    assert "Fisher" in e.method
    larger = pd.DataFrame({"x": ["a"] * 8 + ["b", "c"], "y": ["no"] * 7 + ["yes"] * 3})
    e = execute_analysis(larger, Hypothesis("groups", "x", "y", "test"))
    assert e.p_value is None and not e.p_value_reliable


def test_small_sample_spearman_uses_reproducible_permutation_test():
    df = pd.DataFrame({"x": np.arange(30), "y": np.arange(30)})
    e = execute_analysis(df, Hypothesis("correlation", "x", "y", "test"))
    assert "permutation" in e.method
    assert e.p_value > 0
    assert e.p_value == execute_analysis(df, Hypothesis("correlation", "x", "y", "test")).p_value


def test_generated_chi_square_expected_counts_are_checked():
    evidence = finding(conclusion="A statistically significant association was measured.")
    evidence.method = "Chi-square test"
    evidence.diagnostics = {"expected_min": 0.2}
    run = run_with(evidence)
    review_investigation(frame(), run)
    assert not evidence.p_value_reliable
    assert evidence.review.adjusted_p_value is None
    assert evidence.review.status == "unsupported"


def test_univariate_result_cannot_explain_confounding():
    evidence = finding(conclusion="Purchase frequency explains the association between income and churn.")
    evidence.method = "Chi-square test"
    run = run_with(evidence)
    review_investigation(frame(), run)
    assert any(issue["code"] == "confounding_claim" for issue in evidence.review.issues)


def test_live_discovered_positive_claim_with_p_099_is_not_endorsed():
    evidence = finding(p=0.9900930133474232, conclusion="Age groups are associated with churn rates.")
    run = run_with(evidence)
    review_investigation(frame(), run)
    assert evidence.review.status == "unsupported"
    assert any(issue["code"] == "association_claim" for issue in evidence.review.issues)
    assert evidence.review.original_conclusion == "Age groups are associated with churn rates."
    assert "requires review" in evidence.conclusion
