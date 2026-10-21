"""Trusted statistical tools. No eval, exec, or generated Python on the host."""
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Hypothesis:
    kind: str
    x: str
    y: str
    reason: str
    control: str | None = None

    @property
    def key(self):
        return (self.kind, self.x, self.y, self.control)


@dataclass
class Evidence:
    id: str
    hypothesis: dict
    n: int
    statistic: float | None
    p_value: float | None
    effect: float | None
    conclusion: str
    limitations: list[str]
    table: list[dict]
    code: str
    error: str | None = None


def candidates(df: pd.DataFrame, target: str) -> list[Hypothesis]:
    numeric_target = pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10
    result = []
    for name in df.columns:
        s = df[name]
        if name == target or s.nunique() < 2:
            continue
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if not numeric and (s.nunique() > 20 or s.nunique() == len(df)):
            continue
        kind = 'correlation' if numeric and numeric_target else 'groups'
        if numeric_target and not numeric:
            kind = 'numeric_groups'
        result.append(Hypothesis(kind, name, target, f'Test whether {name} is associated with {target}.'))
        if s.isna().any() and not numeric_target:
            result.append(Hypothesis('missingness', name, target,
                                     f'Check whether missing {name} values carry information about {target}.'))
    return result


def validate(df: pd.DataFrame, h: Hypothesis):
    if h.kind not in {'correlation', 'groups', 'numeric_groups', 'missingness', 'stratified'}:
        raise ValueError('Unsupported analysis tool.')
    for name in [h.x, h.y] + ([h.control] if h.control else []):
        if name not in df.columns:
            raise ValueError(f'Unknown column: {name}')
    if h.x == h.y or (h.control and h.control in {h.x, h.y}):
        raise ValueError('Analysis columns must be distinct.')
    if h.kind == 'stratified' and not h.control:
        raise ValueError('A stratified analysis requires a control column.')
    if h.kind in {'groups', 'missingness', 'stratified'} and df[h.y].nunique() > 20:
        raise ValueError('Categorical target must have at most 20 levels.')


def _group(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 5:
        return pd.qcut(s, 4, duplicates='drop').astype(str).where(s.notna())
    if s.nunique() > 20:
        raise ValueError('Grouping column has more than 20 categories.')
    return s.astype(str).where(s.notna())


def execute_analysis(df: pd.DataFrame, h: Hypothesis, index: int = 1) -> Evidence:
    """Run a fixed tool selected by the planner; return numbers and exact replay code."""
    validate(df, h)
    limitations = ['Exploratory association; this does not establish causation.',
                   'P-values are unadjusted; account for the full investigation when interpreting them.']
    statistic = p = effect = None
    table = []
    if h.kind == 'correlation':
        d = df[[h.x, h.y]].dropna()
        if len(d) < 8 or min(d.nunique()) < 2:
            raise ValueError('Correlation requires at least 8 complete rows and nonconstant columns.')
        result = stats.spearmanr(d[h.x], d[h.y])
        statistic, p = float(result.statistic), float(result.pvalue)
        effect = abs(statistic)
        table = d.head(1000).to_dict('records')
        conclusion = f'Spearman correlation between {h.x} and {h.y}: {statistic:.3f}.'
        n = len(d)
    elif h.kind == 'numeric_groups':
        d = df[[h.x, h.y]].dropna().copy()
        d['group'] = _group(d[h.x])
        groups = [g[h.y].to_numpy() for _, g in d.groupby('group', observed=True)]
        if len(groups) < 2 or min(map(len, groups)) < 5:
            raise ValueError('Need at least two groups with 5 complete observations each.')
        result = stats.kruskal(*groups)
        statistic, p = float(result.statistic), float(result.pvalue)
        effect = max(0., (statistic - len(groups) + 1) / max(1, len(d) - len(groups)))
        table = d.groupby('group', observed=True)[h.y].agg(['count', 'mean', 'median']).reset_index().to_dict('records')
        n = len(d)
        conclusion = f'Compared {h.y} across {len(groups)} {h.x} groups using Kruskal–Wallis.'
    else:
        required = [h.y] + ([] if h.kind == 'missingness' else [h.x]) + ([h.control] if h.control else [])
        d = df.dropna(subset=required).copy()
        d['__group'] = df.loc[d.index, h.x].isna().map({True: 'Missing', False: 'Present'}) if h.kind == 'missingness' else _group(d[h.x])
        if h.kind == 'stratified':
            d['__stratum'] = _group(d[h.control])
        else:
            d['__stratum'] = 'All rows'
        tables, effects, p_values, counts = [], [], [], []
        for label, subset in d.groupby('__stratum', observed=True):
            ct = pd.crosstab(subset['__group'], subset[h.y])
            if min(ct.shape) < 2 or ct.values.sum() < 10:
                continue
            chi, pv, _, expected = stats.chi2_contingency(ct)
            if (expected < 5).any():
                limitations.append(f'Sparse expected counts in {label}; chi-square approximation may be unreliable.')
            v = math.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1)))
            effects.append(v)
            p_values.append(float(pv))
            counts.append(int(ct.values.sum()))
            tables.append((label, ct))
        if not tables:
            raise ValueError('Not enough observations or category variation for this comparison.')
        n = sum(counts)
        effect = float(np.average(effects, weights=counts))
        if h.kind != 'stratified':
            p = p_values[0]
            statistic = float(stats.chi2_contingency(tables[0][1])[0])
        else:
            limitations.append('Stratum-specific comparisons are descriptive, not a causal adjustment or a pooled significance test.')
        for label, ct in tables:
            for group in ct.index:
                total = int(ct.loc[group].sum())
                for category in ct.columns:
                    table.append(dict(stratum=str(label), group=str(group), category=str(category),
                                      count=int(ct.loc[group, category]), total=total,
                                      proportion=float(ct.loc[group, category] / total)))
        conclusion = f'{h.x} vs {h.y}: Cramér’s V = {effect:.3f}'
        if h.control:
            conclusion += f' (weighted within {h.control} strata)'
        conclusion += '.'
    if n < 50:
        limitations.append('Small sample: fewer than 50 usable observations.')
    if n < len(df):
        limitations.append(f'Used {n} of {len(df)} rows; missing values can bias the comparison.')
    values = [v for v in (statistic, p, effect) if v is not None]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('The statistical test returned an undefined result.')
    code = ('from pico.analysis import Hypothesis, execute_analysis\n'
            'from pico.data import load_csv\n'
            'from pathlib import Path\n'
            'df = load_csv(Path("dataset.csv").read_bytes())\n'
            f'result = execute_analysis(df, Hypothesis(**{asdict(h)!r}), {index})\n'
            'print(result)\n')
    return Evidence(f'A{index:02}', asdict(h), n, statistic, p, effect, conclusion,
                    list(dict.fromkeys(limitations)), table, code)
