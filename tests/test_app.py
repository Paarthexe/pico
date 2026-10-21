from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_investigation_and_stale_settings():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    run = app.session_state["run"]
    assert run.target == "churn"
    assert len(run.evidence) == 5
    assert any(e.hypothesis["kind"] == "stratified" for e in run.evidence)
    assert app.session_state["bundle"].startswith(b"PK")
    app.slider[0].set_value(2).run(timeout=30)
    assert any("Settings have changed" in w.value for w in app.warning)


def test_switching_model_or_engine_marks_old_results_stale():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.button[0].click().run(timeout=30)
    engine = next(select for select in app.selectbox if select.label == "Analysis engine")
    engine.select("Generated Python (Docker)").run(timeout=30)
    assert not app.exception
    assert any("Settings have changed" in w.value for w in app.warning)
    assert any(text.label == "Ollama model" for text in app.text_input)


def test_generated_mode_surfaces_missing_docker(monkeypatch):
    from pico.sandbox import SandboxUnavailable

    def unavailable(self):
        raise SandboxUnavailable("Docker fixture unavailable")

    monkeypatch.setattr("pico.sandbox.DockerSandbox.check_available", unavailable)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    next(select for select in app.selectbox if select.label == "Analysis engine").select(
        "Generated Python (Docker)"
    ).run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any("Docker fixture unavailable" in error.value for error in app.error)
    assert "run" not in app.session_state
