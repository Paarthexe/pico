from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_investigation_and_stale_settings():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    run = app.session_state['run']
    assert run.target == 'churn'
    assert len(run.evidence) == 5
    assert any(e.hypothesis['kind'] == 'stratified' for e in run.evidence)
    assert app.session_state['bundle'].startswith(b'PK')
    app.slider[0].set_value(2).run(timeout=30)
    assert any('Settings have changed' in w.value for w in app.warning)
