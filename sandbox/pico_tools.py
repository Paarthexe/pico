"""Container-only conveniences for generated code; Docker provides the isolation."""

import json
from pathlib import Path

import pandas as pd


def load_dataset(path="/input/dataset.csv"):
    return pd.read_csv(path)


def save_result(
    *,
    n,
    conclusion,
    limitations,
    table,
    statistic=None,
    p_value=None,
    effect=None,
    method="",
    assumptions=None,
    diagnostics=None,
):
    if isinstance(table, pd.DataFrame):
        table = table.to_dict(orient="records")
    if not isinstance(table, list) or not 1 <= len(table) <= 1000:
        raise ValueError("Supply a nonempty aggregate evidence table with at most 1000 rows.")
    result = dict(
        n=int(n),
        conclusion=str(conclusion),
        limitations=limitations,
        table=table,
        statistic=statistic,
        p_value=p_value,
        effect=effect,
        method=method,
        assumptions=assumptions or [],
        diagnostics=diagnostics or {},
    )

    def scalar(value):
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    text = json.dumps(result, allow_nan=False, default=scalar)
    if len(text.encode()) > 256 * 1024:
        raise ValueError("Result exceeded 256 KB; aggregate the evidence table.")
    Path("/output/result.json").write_text(text)
