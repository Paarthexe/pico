"""Fixed numeric recipes, independently rebuilt from original complete-case data."""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from pico.models import Evidence

RECIPES = {"pearson_r", "welch_t", "ols_simple"}
CONFIDENCE_LEVEL = 0.95
INDEPENDENCE_ASSUMPTION = (
    "Independent observations; these intervals are not corrected for adaptive selection or multiple testing."
)


@dataclass(frozen=True)
class _Calculation:
    """Host-computed values kept separate from the model's reported evidence."""

    statistic: float
    p_value: float
    effect: float
    effect_name: str
    ci_low: float
    ci_high: float
    table: list[dict]
    assumption: str


def _match(actual: float, supplied: object, label: str) -> None:
    # Tiny p-values need a much tighter absolute tolerance than summary values.
    absolute_tolerance = 1e-300 if label == "p-value" else 1e-12
    if (
        type(supplied) not in {int, float}
        or not math.isfinite(supplied)
        or not math.isclose(actual, supplied, rel_tol=1e-6, abs_tol=absolute_tolerance)
    ):
        raise ValueError(f"Reported {label} does not match independent recomputation.")


def _complete_cases(df: pd.DataFrame, evidence: Evidence) -> pd.DataFrame:
    predictor, outcome = evidence.hypothesis["x"], evidence.hypothesis["y"]
    if predictor == outcome or predictor not in df or outcome not in df:
        raise ValueError("Two distinct original columns are required.")
    specification = evidence.diagnostics
    if specification.get("verification_missing") != "complete_case" or specification.get("verification_bins", 0) != 0:
        raise ValueError("Numeric recipes require complete cases with no binning, filtering or imputation.")
    paired = df[[predictor, outcome]].dropna()
    if len(paired) < 4 or evidence.n != len(paired):
        raise ValueError("At least four complete cases and the exact sample size are required.")
    if not pd.api.types.is_numeric_dtype(paired[outcome]) or not np.isfinite(paired[outcome]).all():
        raise ValueError("Outcome must contain finite original numeric values.")
    return paired


def _welch_comparison(paired: pd.DataFrame, predictor: str, outcome: str) -> _Calculation:
    labels = paired[predictor].astype(str)
    group_names = sorted(labels.unique())
    if len(group_names) != 2 or paired[predictor].nunique() != 2:
        raise ValueError("Welch comparison requires exactly two unambiguous groups.")
    first_group, second_group = (paired.loc[labels == name, outcome].to_numpy(dtype=float) for name in group_names)
    if min(len(first_group), len(second_group)) < 2 or min(first_group.var(ddof=1), second_group.var(ddof=1)) <= 0:
        raise ValueError("Each group needs at least two observations and nonzero variance.")

    # This order defines both the sign of the test and the mean difference.
    result = stats.ttest_ind(second_group, first_group, equal_var=False, alternative="two-sided")
    interval = result.confidence_interval(confidence_level=CONFIDENCE_LEVEL)
    table = [
        dict(group=name, n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)))
        for name, values in zip(group_names, [first_group, second_group])
    ]
    return _Calculation(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        effect=float(second_group.mean() - first_group.mean()),
        effect_name=f"Mean difference ({group_names[1]} minus {group_names[0]})",
        ci_low=float(interval.low),
        ci_high=float(interval.high),
        table=table,
        assumption="Welch t interval: independent groups, finite variance; small-sample inference assumes approximately normal group distributions.",
    )


def _numeric_arrays(paired: pd.DataFrame, predictor: str, outcome: str) -> tuple[np.ndarray, np.ndarray]:
    if not pd.api.types.is_numeric_dtype(paired[predictor]) or not np.isfinite(paired[predictor]).all():
        raise ValueError("Predictor must contain finite original numeric values.")
    predictor_values = paired[predictor].to_numpy(dtype=float)
    outcome_values = paired[outcome].to_numpy(dtype=float)
    if predictor_values.std() == 0 or outcome_values.std() == 0:
        raise ValueError("Constant variables cannot support this recipe.")
    near_constant_threshold = np.finfo(float).eps ** 0.75
    if any(
        values.std() < abs(values.mean()) * near_constant_threshold for values in [predictor_values, outcome_values]
    ):
        raise ValueError("Nearly constant variables are numerically unstable for this recipe.")
    return predictor_values, outcome_values


def _numeric_comparison(paired: pd.DataFrame, predictor: str, outcome: str, recipe: str) -> _Calculation:
    predictor_values, outcome_values = _numeric_arrays(paired, predictor, outcome)
    table = [dict(n=len(paired), x_mean=float(predictor_values.mean()), y_mean=float(outcome_values.mean()))]
    if recipe == "pearson_r":
        result = stats.pearsonr(predictor_values, outcome_values, alternative="two-sided")
        interval = result.confidence_interval(confidence_level=CONFIDENCE_LEVEL)
        return _Calculation(
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            effect=float(result.statistic),
            effect_name="Pearson r",
            ci_low=float(interval.low),
            ci_high=float(interval.high),
            table=table,
            assumption="Pearson exact null p-value and Fisher-z interval assume independent bivariate-normal observations; linearity and outliers require review.",
        )
    if recipe == "ols_simple":
        result = stats.linregress(predictor_values, outcome_values, alternative="two-sided")
        slope = float(result.slope)
        radius = float(stats.t.ppf(0.975, len(predictor_values) - 2) * result.stderr)
        return _Calculation(
            statistic=slope,
            p_value=float(result.pvalue),
            effect=slope,
            effect_name=f"Slope of {outcome} per unit {predictor}",
            ci_low=slope - radius,
            ci_high=slope + radius,
            table=table,
            assumption="Classical OLS slope t interval assumes linearity, independent homoskedastic normal errors; no covariates or cluster adjustment.",
        )
    raise ValueError("Unknown numeric recipe.")


def _match_summary(expected: list[dict], reported: list[dict]) -> None:
    """Check the entire summary rather than accepting a matching p-value alone."""
    if len(reported) != len(expected):
        raise ValueError("Numeric summary table has the wrong number of rows.")
    reported_rows = sorted(reported, key=lambda row: str(row.get("group", "")))
    for expected_row, reported_row in zip(expected, reported_rows):
        if set(expected_row) != set(reported_row):
            raise ValueError("Numeric summary table fields do not match the recipe.")
        for field, expected_value in expected_row.items():
            if isinstance(expected_value, str):
                if reported_row[field] != expected_value:
                    raise ValueError("Group labels do not match.")
            else:
                _match(expected_value, reported_row[field], f"table {field}")


def _match_evidence(calculation: _Calculation, evidence: Evidence) -> None:
    values = [calculation.statistic, calculation.p_value, calculation.effect, calculation.ci_low, calculation.ci_high]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Degenerate data produced a nonfinite estimate or interval.")
    _match(calculation.statistic, evidence.statistic, "statistic")
    _match(calculation.p_value, evidence.p_value, "p-value")
    if evidence.effect is not None:
        _match(calculation.effect, evidence.effect, "effect")
    _match_summary(calculation.table, evidence.table)

    specification = evidence.diagnostics
    if any(field in specification for field in ["ci_low", "ci_high", "confidence_level"]):
        _match(CONFIDENCE_LEVEL, specification.get("confidence_level"), "confidence level")
        _match(calculation.ci_low, specification.get("ci_low"), "CI lower bound")
        _match(calculation.ci_high, specification.get("ci_high"), "CI upper bound")


def verify_numeric(df: pd.DataFrame, evidence: Evidence) -> dict:
    """Rebuild the declared recipe, compare the report, and return audit details."""
    recipe = evidence.diagnostics.get("verification_test")
    try:
        paired = _complete_cases(df, evidence)
        predictor, outcome = evidence.hypothesis["x"], evidence.hypothesis["y"]
        with warnings.catch_warnings():
            warnings.simplefilter("error", stats.ConstantInputWarning)
            warnings.simplefilter("error", stats.NearConstantInputWarning)
            calculation = (
                _welch_comparison(paired, predictor, outcome)
                if recipe == "welch_t"
                else _numeric_comparison(paired, predictor, outcome, recipe)
            )
        _match_evidence(calculation, evidence)
        return dict(
            status="matched",
            recipe=recipe,
            n=len(paired),
            statistic=calculation.statistic,
            p_value=calculation.p_value,
            effect=calculation.effect,
            effect_name=calculation.effect_name,
            ci_low=calculation.ci_low,
            ci_high=calculation.ci_high,
            confidence_level=CONFIDENCE_LEVEL,
            interval_scope="Pointwise, model-based 95% interval; not multiplicity- or selection-adjusted.",
            assumptions=[INDEPENDENCE_ASSUMPTION, calculation.assumption],
            table=calculation.table,
            reason="Sample, summary table, statistic and p-value match independent recomputation; effect and 95% interval computed by the host.",
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
        stats.ConstantInputWarning,
        stats.NearConstantInputWarning,
    ) as exc:
        return dict(status="failed", reason=str(exc))
