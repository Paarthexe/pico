"""Small session-state operations with no Streamlit dependency."""

from collections.abc import MutableMapping


def clear_results(state: MutableMapping) -> None:
    for key in ["run", "bundle", "run_signature"]:
        state.pop(key, None)


def select_dataset(state: MutableMapping, fingerprint: str) -> None:
    if state.get("dataset_id") != fingerprint:
        state["dataset_id"] = fingerprint
        clear_results(state)
        state.pop("target", None)


def settings_signature(target: str, question: str, budget: int, model: str | None, engine: str) -> tuple:
    return target, question, budget, model, engine
