"""Chart rendering and the common chart interface for both execution engines."""

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pico.models import Evidence


def generate_plot(e: Evidence) -> bytes:
    h = e.hypothesis
    table = pd.DataFrame(e.table)
    with plt.rc_context(
        {"axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#121c22", "text.color": "#121c22"}
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        if h["kind"] == "correlation":
            ax.scatter(table[h["x"]], table[h["y"]], s=12, alpha=0.35, color="#017ca8")
            ax.set(xlabel=h["x"], ylabel=h["y"])
        elif h["kind"] == "numeric_groups":
            ax.bar(table["group"].astype(str), table["median"], color="#017ca8")
            ax.set(xlabel=h["x"], ylabel=f"Median {h['y']}")
        else:
            # Each group/stratum sums to 100%; no privileged positive label assumed.
            def short_label(value):
                return value.split(" ", 1)[0] if value.startswith(("Q1 ", "Q2 ", "Q3 ", "Q4 ")) else value

            pivot = table.pivot(index=["stratum", "group"], columns="category", values="proportion").fillna(0)
            pivot.plot.bar(stacked=True, ax=ax, colormap="tab20", width=0.7)
            ax.set_xticklabels(
                [
                    short_label(group) + (" / " + short_label(stratum) if h["control"] else "")
                    for stratum, group in pivot.index
                ]
            )
            ax.set(xlabel=h["x"], ylabel=f"Proportion of {h['y']}", ylim=(0, 1))
            if h["control"]:
                ax.set_xlabel(f"{h['x']} group / {h['control']} stratum")
            ax.legend(title=h["y"], bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_title(f"{e.id} · {h['x']} and {h['y']}", loc="left", pad=16)
        if h["kind"] == "correlation" and e.n > len(table):
            ax.set_title(f"{e.id} · {h['x']} and {h['y']} (first {len(table):,} complete rows)", loc="left", pad=16)
        ax.tick_params(axis="x", labelrotation=35)
        output = BytesIO()
        fig.savefig(output, format="png", dpi=140)
        plt.close(fig)
    return output.getvalue()


def charts_for(evidence: Evidence) -> list[tuple[str, bytes]]:
    if evidence.execution is not None:
        return [
            (f"{evidence.id}-{a.name}", base64.b64decode(a.data_base64, validate=True))
            for a in evidence.execution.artifacts
        ]
    return [(f"{evidence.id}.png", generate_plot(evidence))]
