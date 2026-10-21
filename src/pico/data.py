"""CSV ingestion and compact, serializable data health summaries."""
from io import BytesIO
import numpy as np
import pandas as pd

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 100


def load_csv(raw: bytes) -> pd.DataFrame:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('Choose a nonempty CSV smaller than 20 MB.')
    try:
        df = pd.read_csv(BytesIO(raw), nrows=MAX_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f'Could not read this UTF-8 CSV: {exc}') from exc
    if df.empty or len(df) > MAX_ROWS or len(df.columns) > MAX_COLUMNS:
        raise ValueError('CSV must contain 1–100,000 rows and at most 100 columns.')
    df.columns = [str(c) for c in df.columns]
    return df.replace([np.inf, -np.inf], np.nan)


def profile_dataset(df: pd.DataFrame) -> dict:
    columns = []
    for name in df.columns:
        s = df[name]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        unique = int(s.nunique())
        row = dict(name=name, kind='numeric' if numeric else 'categorical',
                   missing=int(s.isna().sum()), unique=unique,
                   identifier=unique == len(df) and len(df) > 20,
                   constant=unique <= 1)
        if numeric and s.notna().any():
            row.update(mean=float(s.mean()), median=float(s.median()),
                       minimum=float(s.min()), maximum=float(s.max()))
        columns.append(row)
    return dict(rows=len(df), columns=len(df.columns),
                missing_cells=int(df.isna().sum().sum()),
                missing_percent=float(df.isna().sum().sum() / df.size * 100),
                duplicates=int(df.duplicated().sum()), schema=columns)


def demo_dataset() -> pd.DataFrame:
    """Synthetic data: purchase frequency confounds income/churn association."""
    rng = np.random.default_rng(71)
    n = 2400
    purchases = rng.poisson(8, n)
    income = np.maximum(15000, 22000 + purchases * 4400 + rng.normal(0, 17000, n))
    probability = 1 / (1 + np.exp(- (1.5 - .31 * purchases)))
    churn = rng.random(n) < probability
    income[rng.random(n) < np.where(churn, .13, .045)] = np.nan
    return pd.DataFrame(dict(age=rng.integers(18, 76, n), income=income.round(0),
                             purchases=purchases, churn=np.where(churn, 'Yes', 'No'),
                             segment=rng.choice(['Individual', 'Family', 'Business'], n)))
