"""Deterministic statistical critic; model self-assessments are never treated as proof."""

import ast
import math
import re

import numpy as np
import pandas as pd

from pico.models import Evidence, FindingReview, Investigation

ADAPTIVE_CAVEAT = (
    "Holm adjustments describe the reported test family only. Adaptive selection, "
    "unreported tests, and invalid assumptions can still invalidate inference; use independent validation."
)
_IDENTIFIER = re.compile(r"(^id$|(?:^|_)id$|uuid$|^index$|^row_?number$|^customerid$|^userid$)", re.I)
_POST_OUTCOME = re.compile(r"(?:^|_)(?:after|post|future|outcome|label)(?:_|$)", re.I)


def feature_warnings(df: pd.DataFrame, target: str) -> list[dict]:
    warnings = []
    for name in df.columns:
        if name == target:
            continue
        if _IDENTIFIER.search(name):
            warnings.append(dict(column=name, code="identifier", message=f"{name} looks like a row/entity identifier."))
        paired = df[[name, target]].dropna()
        if len(paired) >= max(20, len(df) * 0.8):
            equal = paired[name].eq(paired[target]).all()
            if equal:
                warnings.append(
                    dict(column=name, code="target_copy", message=f"{name} duplicates the observed outcome.")
                )
        if _POST_OUTCOME.search(name) or name.lower().startswith(target.lower() + "_"):
            warnings.append(
                dict(
                    column=name,
                    code="post_outcome",
                    message=f"{name} may encode information recorded after the outcome; verify timing.",
                )
            )
    return warnings


def dataset_warnings(df: pd.DataFrame, target: str) -> list[dict]:
    warnings = feature_warnings(df, target)
    duplicates = int(df.duplicated().sum())
    if duplicates:
        warnings.append(
            dict(column=None, code="duplicates", message=f"{duplicates} duplicate rows may violate independence.")
        )
    counts = df[target].value_counts()
    if 2 <= len(counts) <= 20 and (counts.min() < 10 or counts.max() / counts.sum() > 0.95):
        warnings.append(
            dict(
                column=target,
                code="imbalance",
                message="Outcome has rare classes; estimates and tests may be unstable.",
            )
        )
    return warnings


def holm_adjust(p_values: list[float]) -> list[float]:
    """Step-down Bonferroni with monotonic adjusted values in original order."""
    if any(type(p) not in {int, float} or not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("Holm adjustment requires finite p-values between 0 and 1.")
    if not p_values:
        return []
    order = np.argsort(p_values, kind="stable")
    ranked = np.asarray(p_values, dtype=float)[order]
    adjusted = np.minimum(1.0, np.maximum.accumulate(ranked * np.arange(len(ranked), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def _reported_tests(evidence: Evidence) -> list[tuple[str, float]]:
    if not evidence.p_value_reliable:
        return []
    tests = [("primary", evidence.p_value)] if evidence.p_value is not None else []
    if evidence.execution:
        for index, row in enumerate(evidence.table):
            for key, value in row.items():
                if re.sub(r"[^a-z]", "", key.lower()) == "pvalue" and type(value) in {float, int} and 0 <= value <= 1:
                    tests.append((f"table[{index}].{key}", float(value)))
    # Duplicate reports are counted separately, conservatively, rather than guessing
    # that equal floating-point values identify the same hypothesis.
    return tests


def _mentioned_columns(evidence: Evidence, names: set[str]) -> set[str]:
    mentioned = {evidence.hypothesis.get("x"), evidence.hypothesis.get("control")}
    if evidence.execution:
        try:
            tree = ast.parse(evidence.code)
        except SyntaxError:
            return mentioned
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript):
                mentioned.update(
                    item.value
                    for item in ast.walk(node.slice)
                    if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value in names
                )
    return mentioned


def _asserts(text: str, pattern: str) -> bool:
    for clause in re.split(r"[.!?;]", text.lower()):
        match = re.search(pattern, clause)
        if match and not re.search(r"\b(?:not|no|never|without)\b", clause[max(0, match.start() - 40) : match.start()]):
            return True
    return False


def _review_one(df: pd.DataFrame, run: Investigation, evidence: Evidence, review: FindingReview) -> None:
    def issue(code: str, message: str, severity: str = "warning") -> None:
        review.issues.append(dict(code=code, message=message, severity=severity))

    issue("adaptive_selection", ADAPTIVE_CAVEAT)
    issue(
        "independence", "Independent/exchangeable observations are assumed; the app cannot establish this from a CSV."
    )
    mentioned = _mentioned_columns(evidence, set(df.columns))
    for warning in run.dataset_warnings:
        if warning["column"] in mentioned or warning["column"] is None or warning["code"] == "imbalance":
            blocking = warning["code"] in {"identifier", "target_copy"}
            issue(warning["code"], warning["message"], "blocking" if blocking else "warning")
    if evidence.execution:
        issue(
            "generated_method",
            "Generated code, diagnostics, and assumptions require review; successful execution does not establish correctness.",
        )
        minimum = evidence.diagnostics.get("expected_min", evidence.diagnostics.get("min_expected_count"))
        if type(minimum) in {int, float} and minimum < 5:
            issue(
                "sparse_expected",
                f"Minimum expected cell count is {minimum:.3g}. Aggregate categories or use an appropriate exact test before interpreting a p-value.",
            )
        if not evidence.method or not evidence.assumptions:
            issue("missing_diagnostics", "The generated analysis did not report its method and assumptions.")
    if evidence.n < 50:
        issue("small_sample", "Fewer than 50 usable rows; interpret estimates cautiously.")
    if evidence.n < len(df):
        issue("missingness", f"Used {evidence.n}/{len(df)} observations; missingness/subsetting can bias the result.")
    if not evidence.p_value_reliable:
        issue(
            "unreliable_test",
            "P-values from this analysis are excluded from correction because its test/model assumptions failed.",
            "blocking",
        )
    original = review.original_conclusion
    if _asserts(original, r"\b(?:causes?|caused|causally|proves?|guarantees?|prevents?|drives?)\b"):
        issue(
            "causal_claim",
            "The generated interpretation makes a causal/certain claim unsupported by this exploratory design.",
            "blocking",
        )
    if _asserts(original, r"\b(?:statistically significant|significant (?:association|relationship|difference))\b"):
        if not evidence.p_value_reliable or review.adjusted_p_value is None or review.adjusted_p_value >= 0.05:
            issue(
                "significance_claim",
                "The significance claim is not supported by a reliable primary p-value below 0.05 after adjustment.",
                "blocking",
            )
    if (
        evidence.execution
        and review.adjusted_p_value is not None
        and review.adjusted_p_value >= 0.05
        and _asserts(
            original,
            r"\b(?:is|are|was|were)\s+(?:(?:positively|negatively|strongly|weakly)\s+)?(?:associated|correlated|related)\b",
        )
    ):
        issue(
            "association_claim",
            "The positive association claim is not supported by the adjusted primary p-value. Report the observed estimate and uncertainty instead.",
            "blocking",
        )
    if re.search(r"\bexplain", original.lower()) and "association between" in original.lower() and evidence.execution:
        if not any(word in evidence.method.lower() for word in ["regression", "stratif", "adjusted"]):
            issue(
                "confounding_claim",
                "A univariate comparison cannot establish whether a third variable explains another association.",
                "blocking",
            )
    if re.search(r"\b(?:no (?:association|relationship)|independent of)\b", original.lower()):
        issue("absence_claim", "A nonsignificant result does not establish absence of a relationship.", "blocking")
    review.status = "unsupported" if any(i["severity"] == "blocking" for i in review.issues) else "caution"
    evidence.conclusion = (
        f"{evidence.id} produced numerical results, but its interpretation requires review; "
        "see the evidence table and flagged original conclusion."
        if review.status == "unsupported"
        else original
    )


def review_investigation(df: pd.DataFrame, run: Investigation) -> None:
    """Recompute after every finding so subsequent planning sees current critiques."""
    run.dataset_warnings = dataset_warnings(df, run.target)
    tests = []
    for evidence in run.evidence:
        if evidence.execution:
            diagnostics = evidence.diagnostics
            failed_fit = diagnostics.get("converged") is False or diagnostics.get("fit_converged") is False
            failed_fit |= bool(
                re.search(r"convergencewarning|perfectseparation|singular matrix", evidence.execution.stderr, re.I)
            )
            if "chi" in evidence.method.lower():
                expected_min = diagnostics.get("expected_min", diagnostics.get("min_expected_count"))
                if type(expected_min) in {int, float} and expected_min < 5:
                    failed_fit = True
            if failed_fit:
                evidence.p_value_reliable = False
        original = evidence.review.original_conclusion if evidence.review else evidence.conclusion
        evidence.review = FindingReview(original)
        tests.extend((evidence, label, p) for label, p in _reported_tests(evidence))
    adjusted = holm_adjust([p for _, _, p in tests])
    for (evidence, label, raw), value in zip(tests, adjusted):
        evidence.review.adjusted_tests.append(dict(test=label, raw_p_value=raw, adjusted_p_value=value))
        if label == "primary":
            evidence.review.adjusted_p_value = value
    run.multiplicity = dict(
        method="Holm",
        family_size=len(tests),
        alpha=0.05,
        scope="Reliable primary p-values and explicitly named table p-values across this run; repeated reports counted separately.",
        limitation=ADAPTIVE_CAVEAT,
    )
    for evidence in run.evidence:
        evidence.review.family_size = len(tests)
        _review_one(df, run, evidence, evidence.review)
