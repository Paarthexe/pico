from dataclasses import asdict
from io import BytesIO
import json
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from pico.analysis import Hypothesis, execute_analysis
from pico.agent import investigate
from pico.data import demo_dataset, load_csv, profile_dataset
from pico.report import export_bundle


def test_ingestion_and_health():
    df = load_csv(b'a,b\n1,x\n1,x\n,\n')
    p = profile_dataset(df)
    assert p['duplicates'] == 1
    assert p['missing_cells'] == 2
    with pytest.raises(ValueError):
        load_csv(b'')
    with pytest.raises(ValueError):
        load_csv(b'column\n')


def test_numeric_correlation_and_incomplete_rows():
    df = pd.DataFrame({'x': list(range(30)) + [None], 'y': list(range(31))})
    e = execute_analysis(df, Hypothesis('correlation', 'x', 'y', 'test'))
    assert e.n == 30
    assert e.statistic == pytest.approx(1)
    assert any('missing values' in x for x in e.limitations)


def test_missingness_retains_missing_predictor_rows():
    df = pd.DataFrame({'x': [None] * 30 + list(range(30)), 'y': ['yes'] * 30 + ['no'] * 30})
    e = execute_analysis(df, Hypothesis('missingness', 'x', 'y', 'test'))
    assert e.n == 60
    assert {r['group'] for r in e.table} == {'Missing', 'Present'}
    assert e.effect > .9


def test_demo_replans_and_stratifies_without_causal_claim():
    r = investigate(demo_dataset(), 'churn')
    assert len(r.evidence) <= 5
    assert any(e.hypothesis['kind'] == 'stratified' for e in r.evidence)
    assert any(a['stage'] == 'Replan' for a in r.activity)
    assert len({tuple(e.hypothesis[k] for k in ('kind', 'x', 'y', 'control')) for e in r.evidence}) == len(r.evidence)
    for e in r.evidence:
        assert e.n > 0 and e.code and e.limitations
    json.dumps(asdict(r), allow_nan=False)


def test_budget_and_constant_target():
    with pytest.raises(ValueError):
        investigate(demo_dataset(), 'churn', budget=6)
    with pytest.raises(ValueError):
        investigate(pd.DataFrame({'a': [1, 1]}), 'a')
    assert len(investigate(demo_dataset(), 'churn', budget=1).evidence) == 1


def test_invalid_model_choice_falls_back(monkeypatch):
    def fail(*args):
        raise ValueError('bad model response')
    monkeypatch.setattr('pico.agent.ollama_choice', fail)
    run = investigate(demo_dataset(), 'churn', budget=1, model='invalid')
    assert len(run.evidence) == 1
    assert 'fallback' in run.mode
    assert any(a['stage'] == 'Planner fallback' for a in run.activity)


def test_tools_reject_unknown_columns_and_sparse_comparison():
    with pytest.raises(ValueError, match='Unknown column'):
        execute_analysis(demo_dataset(), Hypothesis('groups', 'absent', 'churn', 'test'))
    with pytest.raises(ValueError):
        execute_analysis(pd.DataFrame({'x': [1, 2], 'y': ['a', 'b']}), Hypothesis('groups', 'x', 'y', 'test'))


def test_export_contains_linked_evidence_and_replay():
    run = investigate(demo_dataset(), 'churn', budget=1)
    with ZipFile(BytesIO(export_bundle(run))) as z:
        assert {'report.md', 'evidence.json', 'investigation.ipynb', 'charts/A01.png', 'code/A01.py'} <= set(z.namelist())
        assert z.read('charts/A01.png').startswith(b'\x89PNG')
        nb = json.loads(z.read('investigation.ipynb'))
        assert nb['nbformat'] == 4
        compile(z.read('code/A01.py'), '<replay>', 'exec')
        assert 'dataset.csv' not in z.namelist()
