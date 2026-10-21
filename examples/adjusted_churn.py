"""Free-form analysis example for Pico's synthetic customer data.

Run only with DockerSandbox, as shown in README.md. Paths belong to the container.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

frame = pd.read_csv("/input/dataset.csv").dropna(subset=["income", "purchases", "churn"])
features = frame[["income", "purchases"]].copy()
features["income"] /= 10_000
outcome = frame["churn"].eq("Yes").astype(int)
fit = sm.Logit(outcome, sm.add_constant(features)).fit(disp=False)
intervals = fit.conf_int()
table = [
    dict(
        term=term,
        odds_ratio=float(np.exp(fit.params[term])),
        ci_low=float(np.exp(intervals.loc[term, 0])),
        ci_high=float(np.exp(intervals.loc[term, 1])),
        p_value=float(fit.pvalues[term]),
    )
    for term in ["income", "purchases"]
]
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for row in table:
    ax.errorbar(
        row["odds_ratio"],
        row["term"],
        xerr=[[row["odds_ratio"] - row["ci_low"]], [row["ci_high"] - row["odds_ratio"]]],
        fmt="o",
    )
ax.axvline(1, color="gray", linestyle="--")
ax.set(
    xscale="log",
    xlabel="Adjusted odds ratio (95% confidence interval)",
    title="Income per 10,000 units and purchases versus churn",
)
fig.savefig("/output/adjusted_churn.png", dpi=120)
plt.close(fig)
Path("/output/result.json").write_text(
    json.dumps(
        dict(
            n=len(frame),
            statistic=float(fit.tvalues["income"]),
            p_value=float(fit.pvalues["income"]),
            effect=float(np.exp(fit.params["income"])),
            table=table,
            conclusion=f"Income odds ratio per 10,000 units, controlling for purchases: {np.exp(fit.params['income']):.3f}.",
            limitations=[
                "Associational logistic regression on synthetic data, not a causal estimate.",
                "Complete-case analysis may be biased by missingness.",
                "Model assumptions, multiple testing, and confidence interval coverage need review.",
            ],
        )
    )
)
print(f"Fitted logistic regression on {len(frame)} complete observations.")
