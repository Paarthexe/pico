"""User-supplied study design declarations and deterministic design checks."""

import pandas as pd

from pico.models import StudyContext


def validate_context(df: pd.DataFrame, target: str, context: StudyContext) -> None:
    if not isinstance(context, StudyContext):
        raise ValueError("Study context must be a StudyContext record.")
    if context.excluded_columns - set(df.columns):
        raise ValueError("Study context references a column absent from the dataset.")
    if target in context.excluded_columns:
        raise ValueError("The outcome cannot also be a subject, timestamp or excluded predictor.")
    role_columns = [column for column in [context.subject_id, context.observation_time, context.outcome_time] if column]
    if len(set(role_columns)) != len(role_columns):
        raise ValueError("Subject and timestamp roles must use distinct columns.")


def _warning(column: str | None, code: str, message: str, *, blocking: bool = False) -> dict:
    return dict(column=column, code=code, message=message, blocking=blocking)


def _subject_warnings(subject: pd.Series) -> list[dict]:
    warnings = []
    repeated_rows = int(subject.dropna().duplicated(keep=False).sum())
    if repeated_rows:
        warnings.append(
            _warning(
                None,
                "repeated_subjects",
                f"{repeated_rows} rows belong to repeated subjects. Independent-row tests and intervals are suppressed; use a repeated-measures or cluster-aware method.",
                blocking=True,
            )
        )
    if subject.isna().any():
        warnings.append(
            _warning(
                None,
                "missing_subject_ids",
                "Missing subject identifiers prevent checking independence for all rows; independent-row inference is suppressed.",
                blocking=True,
            )
        )
    return warnings


def _parse_timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True, format="mixed")


def _timestamp_column_warning(values: pd.Series, column: str) -> dict | None:
    if pd.api.types.is_numeric_dtype(values) or values.isna().any():
        return _warning(
            None,
            "uncertain_timestamps",
            f"{column} has missing or numeric timestamps. Convert to explicit ISO dates before chronology checks; inference is suppressed.",
            blocking=True,
        )
    if _parse_timestamps(values).isna().any():
        return _warning(
            None,
            "uncertain_timestamps",
            f"{column} contains unparseable timestamps; chronology is unresolved and inference is suppressed.",
            blocking=True,
        )
    return None


def _chronology_warnings(observed_values: pd.Series, outcome_values: pd.Series) -> list[dict]:
    # Numeric timestamps have ambiguous units; never guess nanoseconds.
    if any(pd.api.types.is_numeric_dtype(values) for values in [observed_values, outcome_values]):
        return []
    observed = _parse_timestamps(observed_values)
    outcome = _parse_timestamps(outcome_values)
    warnings = []
    if observed.isna().any() or outcome.isna().any():
        warnings.append(
            _warning(
                None,
                "uncertain_timestamps",
                "Some timestamps cannot be parsed; chronology is unresolved and inference is suppressed.",
                blocking=True,
            )
        )
    nonpreceding_rows = int((observed >= outcome).sum())
    if nonpreceding_rows:
        warnings.append(
            _warning(
                None,
                "post_outcome_observations",
                f"{nonpreceding_rows} observation timestamps are at or after the outcome. Predictor availability is unresolved; inference is suppressed.",
                blocking=True,
            )
        )
    return warnings


def _temporal_warnings(df: pd.DataFrame, context: StudyContext) -> list[dict]:
    if not (context.observation_time or context.outcome_time):
        return []
    warnings = [
        _warning(
            None,
            "temporal_design",
            "Timestamp context is declared. These recipes do not adjust for temporal dependence or establish prediction-time availability.",
        )
    ]
    for column in [context.observation_time, context.outcome_time]:
        if column:
            warning = _timestamp_column_warning(df[column], column)
            if warning is not None:
                warnings.append(warning)
    if context.observation_time and context.outcome_time:
        warnings.extend(_chronology_warnings(df[context.observation_time], df[context.outcome_time]))
    return warnings


def study_warnings(df: pd.DataFrame, context: StudyContext) -> list[dict]:
    """Collect declared exclusions, subject checks, and temporal checks in order."""
    warnings = [
        _warning(
            column,
            "declared_post_outcome",
            f"{column} was declared post-outcome and is excluded from eligible predictors.",
            blocking=True,
        )
        for column in context.post_outcome_columns
    ]
    if context.subject_id:
        warnings.extend(_subject_warnings(df[context.subject_id]))
    warnings.extend(_temporal_warnings(df, context))
    return warnings
