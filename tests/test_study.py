import numpy as np
import pandas as pd
import pytest

from pico.agent import investigate
from pico.data import demo_dataset
from pico.models import StudyContext
from pico.study import study_warnings, validate_context
from pico.ui.state import select_dataset, settings_signature


def test_declared_post_outcome_and_metadata_excluded_from_catalog_and_followups():
    df = demo_dataset().assign(subject=np.arange(2400), future_score=np.arange(2400))
    run = investigate(
        df,
        "churn",
        budget=5,
        study_context=StudyContext(subject_id="subject", post_outcome_columns=("income", "future_score")),
    )
    for e in run.evidence:
        assert e.hypothesis["x"] not in {"subject", "income", "future_score"}
        assert e.hypothesis.get("control") not in {"subject", "income", "future_score"}
    assert run.study_context.subject_id == "subject"


def test_context_changes_invalidate_signature_and_dataset_changes_clear_widgets():
    assert settings_signature("y", "", 2, None, "catalog") != settings_signature(
        "y", "", 2, None, "catalog", StudyContext(subject_id="id")
    )
    state = dict(
        dataset_id="old",
        study_subject="id",
        study_observed="time",
        study_outcome="later",
        study_post=["leak"],
        run="old",
    )
    select_dataset(state, "new")
    assert not any(k.startswith("study_") for k in state)
    assert "run" not in state


@pytest.mark.parametrize(
    "context",
    [
        StudyContext(subject_id="missing"),
        StudyContext(subject_id="y"),
        StudyContext(subject_id="id", observation_time="id"),
        StudyContext(post_outcome_columns=("y",)),
    ],
)
def test_invalid_roles_are_rejected(context):
    with pytest.raises(ValueError):
        validate_context(pd.DataFrame(dict(id=[1, 2], y=[0, 1])), "y", context)


def test_timestamp_order_missing_and_invalid_dates_suppress_inference():
    df = pd.DataFrame(dict(seen=["2026-01-01", "2026-01-03"], event=["2026-01-02", "2026-01-02"]))
    context = StudyContext(observation_time="seen", outcome_time="event")
    warnings = study_warnings(df, context)
    assert any(w["code"] == "post_outcome_observations" and w["blocking"] for w in warnings)
    df.loc[0, "seen"] = "invalid"
    assert any(w["code"] == "uncertain_timestamps" and w["blocking"] for w in study_warnings(df, context))
    df["seen"] = [1, 2]
    assert any(w["code"] == "uncertain_timestamps" and w["blocking"] for w in study_warnings(df, context))


def test_valid_chronology_does_not_prove_temporal_independence():
    df = pd.DataFrame(dict(seen=["2026-01-01", "2026-01-02"], event=["2026-01-03", "2026-01-03"]))
    warnings = study_warnings(df, StudyContext(observation_time="seen", outcome_time="event"))
    assert any(w["code"] == "temporal_design" for w in warnings)
    assert not any(w.get("blocking") for w in warnings)


def test_missing_and_repeated_subject_ids_are_blocking():
    warnings = study_warnings(pd.DataFrame({"subject": ["a", "a", None]}), StudyContext(subject_id="subject"))
    assert {"repeated_subjects", "missing_subject_ids"} <= {w["code"] for w in warnings if w.get("blocking")}
