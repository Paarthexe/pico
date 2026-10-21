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
    roles = [c for c in [context.subject_id, context.observation_time, context.outcome_time] if c]
    if len(set(roles)) != len(roles):
        raise ValueError("Subject and timestamp roles must use distinct columns.")


def study_warnings(df: pd.DataFrame, context: StudyContext) -> list[dict]:
    warnings = []

    def add(column, code, message, blocking=False):
        warnings.append(dict(column=column, code=code, message=message, blocking=blocking))

    for column in context.post_outcome_columns:
        add(
            column,
            "declared_post_outcome",
            f"{column} was declared post-outcome and is excluded from eligible predictors.",
            True,
        )
    if context.subject_id:
        subject = df[context.subject_id]
        repeated = int(subject.dropna().duplicated(keep=False).sum())
        if repeated:
            add(
                None,
                "repeated_subjects",
                f"{repeated} rows belong to repeated subjects. Independent-row tests and intervals are suppressed; use a repeated-measures or cluster-aware method.",
                True,
            )
        if subject.isna().any():
            add(
                None,
                "missing_subject_ids",
                "Missing subject identifiers prevent checking independence for all rows; independent-row inference is suppressed.",
                True,
            )
    if context.observation_time or context.outcome_time:
        add(
            None,
            "temporal_design",
            "Timestamp context is declared. These recipes do not adjust for temporal dependence or establish prediction-time availability.",
        )
        for column in [context.observation_time, context.outcome_time]:
            if column and (pd.api.types.is_numeric_dtype(df[column]) or df[column].isna().any()):
                add(
                    None,
                    "uncertain_timestamps",
                    f"{column} has missing or numeric timestamps. Convert to explicit ISO dates before chronology checks; inference is suppressed.",
                    True,
                )
            elif column and pd.to_datetime(df[column], errors="coerce", utc=True, format="mixed").isna().any():
                add(
                    None,
                    "uncertain_timestamps",
                    f"{column} contains unparseable timestamps; chronology is unresolved and inference is suppressed.",
                    True,
                )
        if context.observation_time and context.outcome_time:
            # Numeric timestamps have ambiguous units; never guess nanoseconds.
            columns = [context.observation_time, context.outcome_time]
            if not any(pd.api.types.is_numeric_dtype(df[c]) for c in columns):
                observed = pd.to_datetime(df[columns[0]], errors="coerce", utc=True, format="mixed")
                outcome = pd.to_datetime(df[columns[1]], errors="coerce", utc=True, format="mixed")
                if observed.isna().any() or outcome.isna().any():
                    add(
                        None,
                        "uncertain_timestamps",
                        "Some timestamps cannot be parsed; chronology is unresolved and inference is suppressed.",
                        True,
                    )
                count = int((observed >= outcome).sum())
                if count:
                    add(
                        None,
                        "post_outcome_observations",
                        f"{count} observation timestamps are at or after the outcome. Predictor availability is unresolved; inference is suppressed.",
                        True,
                    )
    return warnings
