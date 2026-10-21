"""Trusted catalog executor. Each statistical method has one focused implementation."""
from dataclasses import asdict, dataclass
import math
import numpy as np
import pandas as pd
from scipy import stats
from pico.config import EXPLORATORY_CAVEAT, MULTIPLE_TESTING_CAVEAT
from pico.models import Evidence, Hypothesis
from pico.planning import candidates  # Public compatibility import.


@dataclass
class _Summary:
    n: int
    statistic: float | None
    p_value: float | None
    effect: float | None
    conclusion: str
    table: list[dict]


def validate(df: pd.DataFrame, h: Hypothesis) -> None:
    if h.kind not in _METHODS:
        raise ValueError('Unsupported catalog analysis tool.')
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
        bins = pd.qcut(s, 4, duplicates='drop')
        bins = bins.cat.rename_categories([f'Q{i + 1} {interval}'
                                          for i, interval in enumerate(bins.cat.categories)])
        return bins.astype(str).where(s.notna())
    if s.nunique() > 20:
        raise ValueError('Grouping column has more than 20 categories.')
    return s.astype(str).where(s.notna())


def _correlation(df: pd.DataFrame, h: Hypothesis, limitations: list[str]) -> _Summary:
    d = df[[h.x, h.y]].dropna()
    if len(d) < 8 or min(d.nunique()) < 2:
        raise ValueError('Correlation requires at least 8 complete rows and nonconstant columns.')
    result = stats.spearmanr(d[h.x], d[h.y])
    statistic, p = float(result.statistic), float(result.pvalue)
    return _Summary(len(d), statistic, p, abs(statistic),
                    f'Spearman correlation between {h.x} and {h.y}: {statistic:.3f}.',
                    d.head(1000).to_dict('records'))


def _numeric_groups(df: pd.DataFrame, h: Hypothesis, limitations: list[str]) -> _Summary:
    # Fresh internal names avoid clobbering a user column called "group".
    d = pd.DataFrame({'value': df[h.y], 'category': _group(df[h.x])}).dropna()
    groups = [g['value'].to_numpy() for _, g in d.groupby('category', observed=True)]
    if len(groups) < 2 or min(map(len, groups)) < 5:
        raise ValueError('Need at least two groups with 5 complete observations each.')
    result = stats.kruskal(*groups)
    statistic, p = float(result.statistic), float(result.pvalue)
    effect = max(0., (statistic - len(groups) + 1) / max(1, len(d) - len(groups)))
    table = (d.groupby('category', observed=True)['value'].agg(['count', 'mean', 'median'])
             .rename_axis('group').reset_index().to_dict('records'))
    return _Summary(len(d), statistic, p, effect,
                    f'Compared {h.y} across {len(groups)} {h.x} groups using Kruskal–Wallis.', table)


def _categorical(df: pd.DataFrame, h: Hypothesis, limitations: list[str]) -> _Summary:
    groups = df[h.x].isna().map({True: 'Missing', False: 'Present'}) if h.kind == 'missingness' else _group(df[h.x])
    strata = _group(df[h.control]) if h.control else pd.Series('All rows', index=df.index)
    d = pd.DataFrame({'outcome': df[h.y], 'group': groups, 'stratum': strata}).dropna()
    tables, effects, counts = [], [], []
    statistic = p = None
    for label, subset in d.groupby('stratum', observed=True):
        ct = pd.crosstab(subset['group'], subset['outcome'])
        if min(ct.shape) < 2 or ct.values.sum() < 10:
            continue
        chi, pv, _, expected = stats.chi2_contingency(ct)
        if (expected < 5).any():
            limitations.append(f'Sparse expected counts in {label}; chi-square approximation may be unreliable.')
        effects.append(math.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1))))
        counts.append(int(ct.values.sum()))
        tables.append((label, ct))
        if h.kind != 'stratified':
            statistic, p = float(chi), float(pv)
    if not tables:
        raise ValueError('Not enough observations or category variation for this comparison.')
    if h.kind == 'stratified':
        limitations.append('Stratum-specific comparisons are descriptive, not a causal adjustment or a pooled significance test.')
    effect = float(np.average(effects, weights=counts))
    table = []
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
    return _Summary(sum(counts), statistic, p, effect, conclusion + '.', table)


_METHODS = {'correlation': _correlation, 'numeric_groups': _numeric_groups,
            'groups': _categorical, 'missingness': _categorical, 'stratified': _categorical}


def execute_analysis(df: pd.DataFrame, h: Hypothesis, index: int = 1) -> Evidence:
    validate(df, h)
    limitations = [EXPLORATORY_CAVEAT, MULTIPLE_TESTING_CAVEAT]
    result = _METHODS[h.kind](df, h, limitations)
    if result.n < 50:
        limitations.append('Small sample: fewer than 50 usable observations.')
    if result.n < len(df):
        limitations.append(f'Used {result.n} of {len(df)} rows; missing values can bias the comparison.')
    values = [v for v in (result.statistic, result.p_value, result.effect) if v is not None]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('The statistical test returned an undefined result.')
    code = ('from pico.analysis import Hypothesis, execute_analysis\n'
            'from pico.data import load_csv\nfrom pathlib import Path\n'
            'df = load_csv(Path("dataset.csv").read_bytes())\n'
            f'result = execute_analysis(df, Hypothesis(**{asdict(h)!r}), {index})\nprint(result)\n')
    return Evidence(f'A{index:02}', asdict(h), result.n, result.statistic, result.p_value,
                    result.effect, result.conclusion, list(dict.fromkeys(limitations)), result.table, code)
