"""Fixed numeric recipes, independently rebuilt from original complete-case data."""

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from pico.models import Evidence

RECIPES = {"pearson_r", "welch_t", "ols_simple"}


def _match(actual, supplied, label):
    if (
        type(supplied) not in {int, float}
        or not math.isfinite(supplied)
        or not math.isclose(actual, supplied, rel_tol=1e-6, abs_tol=1e-12 if label != "p-value" else 1e-300)
    ):
        raise ValueError(f"Reported {label} does not match independent recomputation.")


def verify_numeric(df: pd.DataFrame, evidence: Evidence) -> dict:
    spec = evidence.diagnostics
    recipe = spec.get("verification_test")
    try:
        x, y = evidence.hypothesis["x"], evidence.hypothesis["y"]
        if x == y or x not in df or y not in df:
            raise ValueError("Two distinct original columns are required.")
        if spec.get("verification_missing") != "complete_case" or spec.get("verification_bins", 0) != 0:
            raise ValueError("Numeric recipes require complete cases with no binning, filtering or imputation.")
        paired = df[[x, y]].dropna()
        if len(paired) < 4 or evidence.n != len(paired):
            raise ValueError("At least four complete cases and the exact sample size are required.")
        if not pd.api.types.is_numeric_dtype(paired[y]) or not np.isfinite(paired[y]).all():
            raise ValueError("Outcome must contain finite original numeric values.")
        assumptions = [
            "Independent observations; these intervals are not corrected for adaptive selection or multiple testing."
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error", stats.ConstantInputWarning)
            warnings.simplefilter("error", stats.NearConstantInputWarning)
            if recipe == "welch_t":
                labels = paired[x].astype(str)
                groups = sorted(labels.unique())
                if len(groups) != 2 or paired[x].nunique() != 2:
                    raise ValueError("Welch comparison requires exactly two unambiguous groups.")
                a, b = (paired.loc[labels == g, y].to_numpy(dtype=float) for g in groups)
                if min(len(a), len(b)) < 2 or min(a.var(ddof=1), b.var(ddof=1)) <= 0:
                    raise ValueError("Each group needs at least two observations and nonzero variance.")
                result = stats.ttest_ind(b, a, equal_var=False, alternative="two-sided")
                statistic, p = float(result.statistic), float(result.pvalue)
                interval = result.confidence_interval(confidence_level=0.95)
                effect, effect_name = float(b.mean() - a.mean()), f"Mean difference ({groups[1]} minus {groups[0]})"
                table = [
                    dict(group=g, n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)))
                    for g, v in zip(groups, [a, b])
                ]
                assumptions.append(
                    "Welch t interval: independent groups, finite variance; small-sample inference assumes approximately normal group distributions."
                )
            else:
                if not pd.api.types.is_numeric_dtype(paired[x]) or not np.isfinite(paired[x]).all():
                    raise ValueError("Predictor must contain finite original numeric values.")
                a, b = paired[x].to_numpy(dtype=float), paired[y].to_numpy(dtype=float)
                if a.std() == 0 or b.std() == 0:
                    raise ValueError("Constant variables cannot support this recipe.")
                if any(v.std() < abs(v.mean()) * np.finfo(float).eps ** 0.75 for v in [a, b]):
                    raise ValueError("Nearly constant variables are numerically unstable for this recipe.")
                table = [dict(n=len(paired), x_mean=float(a.mean()), y_mean=float(b.mean()))]
                if recipe == "pearson_r":
                    result = stats.pearsonr(a, b, alternative="two-sided")
                    statistic, p, effect = float(result.statistic), float(result.pvalue), float(result.statistic)
                    interval = result.confidence_interval(confidence_level=0.95)
                    effect_name = "Pearson r"
                    assumptions.append(
                        "Pearson exact null p-value and Fisher-z interval assume independent bivariate-normal observations; linearity and outliers require review."
                    )
                elif recipe == "ols_simple":
                    result = stats.linregress(a, b, alternative="two-sided")
                    statistic, p, effect = float(result.slope), float(result.pvalue), float(result.slope)
                    radius = float(stats.t.ppf(0.975, len(a) - 2) * result.stderr)
                    interval = (effect - radius, effect + radius)
                    effect_name = f"Slope of {y} per unit {x}"
                    assumptions.append(
                        "Classical OLS slope t interval assumes linearity, independent homoskedastic normal errors; no covariates or cluster adjustment."
                    )
                else:
                    raise ValueError("Unknown numeric recipe.")
        low, high = map(float, interval)
        if not all(math.isfinite(v) for v in [statistic, p, effect, low, high]):
            raise ValueError("Degenerate data produced a nonfinite estimate or interval.")
        _match(statistic, evidence.statistic, "statistic")
        _match(p, evidence.p_value, "p-value")
        if evidence.effect is not None:
            _match(effect, evidence.effect, "effect")
        # The entire declared summary is checked, not just a matching p-value.
        if len(evidence.table) != len(table):
            raise ValueError("Numeric summary table has the wrong number of rows.")
        reported = sorted(evidence.table, key=lambda row: str(row.get("group", "")))
        for actual, supplied in zip(table, reported):
            if set(actual) != set(supplied):
                raise ValueError("Numeric summary table fields do not match the recipe.")
            for key, value in actual.items():
                if isinstance(value, str):
                    if supplied[key] != value:
                        raise ValueError("Group labels do not match.")
                else:
                    _match(value, supplied[key], f"table {key}")
        if any(key in spec for key in ["ci_low", "ci_high", "confidence_level"]):
            _match(0.95, spec.get("confidence_level"), "confidence level")
            _match(low, spec.get("ci_low"), "CI lower bound")
            _match(high, spec.get("ci_high"), "CI upper bound")
        return dict(
            status="matched",
            recipe=recipe,
            n=len(paired),
            statistic=statistic,
            p_value=p,
            effect=effect,
            effect_name=effect_name,
            ci_low=low,
            ci_high=high,
            confidence_level=0.95,
            interval_scope="Pointwise, model-based 95% interval; not multiplicity- or selection-adjusted.",
            assumptions=assumptions,
            table=table,
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
