"""Bounded, trusted recomputation. Never interprets or executes model source.

The declaration selects a fixed recipe, not arbitrary transformations. A match
verifies this calculation only, not study design, causality, or all analysis code.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from pico.models import Evidence


def verify_contingency(df: pd.DataFrame, evidence: Evidence) -> dict:
    spec = evidence.diagnostics
    test = spec.get("verification_test")
    if test not in {"pearson_chi2", "fisher_exact"}:
        return dict(status="unverified", reason="No supported independent verification recipe was supplied.")
    try:
        x, y = evidence.hypothesis["x"], evidence.hypothesis["y"]
        if x == y or x not in df or y not in df:
            raise ValueError("Verification requires two distinct original columns.")
        if spec.get("verification_missing") != "complete_case":
            raise ValueError("Only pairwise complete-case analysis is supported; imputation/subsetting is unverified.")
        paired = df[[x, y]].dropna().copy()
        bins = spec.get("verification_bins", 0)
        if type(bins) is not int or bins not in {0, 4}:
            raise ValueError("Only original categories or four quantile bins are supported.")
        if bins:
            if not pd.api.types.is_numeric_dtype(paired[x]):
                raise ValueError("Quantile binning requires a numeric predictor.")
            if not np.isfinite(paired[x]).all():
                raise ValueError("Infinite values cannot be quantile binned.")
            paired[x] = pd.qcut(paired[x], 4, labels=False)
        # Bound trusted-host computation before constructing a cross-tabulation.
        if not 2 <= paired[x].nunique() <= 20 or not 2 <= paired[y].nunique() <= 20:
            raise ValueError("Verification requires 2–20 observed categories per column.")
        cardinalities = paired.nunique()
        paired = paired.astype(str)
        if not paired.nunique().equals(cardinalities):
            raise ValueError("Distinct categories collapse to the same text label.")
        counts = pd.crosstab(paired[x], paired[y])
        if counts.index.has_duplicates or counts.columns.has_duplicates:
            raise ValueError("Ambiguous category labels.")
        table = [dict(group=g, category=c, count=int(counts.loc[g, c])) for g in counts.index for c in counts.columns]
        reported = {}
        for row in evidence.table:
            if set(row) != {"group", "category", "count"}:
                raise ValueError("Verified evidence requires flat group/category/count rows.")
            key = (str(row["group"]), str(row["category"]))
            if key in reported or type(row["count"]) is not int or row["count"] < 0:
                raise ValueError("Counts must be nonnegative integers with unique category pairs.")
            reported[key] = row["count"]
        expected = {(r["group"], r["category"]): r["count"] for r in table}
        if reported != expected:
            raise ValueError("Reported counts do not match counts rebuilt from the original data.")
        if evidence.n != len(paired):
            raise ValueError("Reported sample size does not match the complete-case sample.")
        minimum = float(stats.contingency.expected_freq(counts).min())
        if test == "pearson_chi2":
            if minimum < 5:
                raise ValueError("Recomputed expected counts are below 5; Pearson inference is not accepted.")
            result = stats.chi2_contingency(counts, correction=False)
        else:
            if counts.shape != (2, 2):
                raise ValueError("This Fisher verification recipe requires a 2×2 table.")
            result = stats.fisher_exact(counts, alternative="two-sided")
        for label, actual, supplied in [
            ("p-value", float(result.pvalue), evidence.p_value),
            ("statistic", float(result.statistic), evidence.statistic),
        ]:
            if not math.isfinite(actual):
                if supplied is not None:
                    raise ValueError(f"The {label} is undefined/infinite and must not be reported as finite.")
            elif supplied is None or not math.isclose(actual, supplied, rel_tol=1e-6, abs_tol=1e-300):
                raise ValueError(f"Reported {label} does not match independent recomputation.")
        return dict(
            status="matched",
            recipe=test,
            n=len(paired),
            p_value=float(result.pvalue),
            statistic=float(result.statistic) if math.isfinite(result.statistic) else None,
            min_expected_count=minimum,
            table=table,
            reason="Counts, sample size, statistic and p-value match independent recomputation; study assumptions remain unverified.",
        )
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return dict(status="failed", reason=str(exc))
