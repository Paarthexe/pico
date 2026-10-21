"""Generated planning, feedback, evidence provenance, and safe exports."""

import json
from dataclasses import asdict
from io import BytesIO
from zipfile import ZipFile

import pytest

from pico.agent import investigate
from pico.data import demo_dataset, profile_dataset
from pico.models import ExecutionRecord, Investigation, PythonProposal
from pico.ollama import PlannerError
from pico.planning import PythonPlanner
from pico.report import export_bundle, notebook
from pico.sandbox import SandboxUnavailable

RESULT = dict(
    n=100,
    conclusion="Executed test found a mean of 12.",
    limitations=["Fixture only."],
    table=[{"mean": 12}],
    statistic=12,
    p_value=None,
    effect=None,
)


class FakeExecutor:
    def __init__(self, statuses=("ok",)):
        self.statuses = iter(statuses)
        self.checked = False
        self.calls = []

    def check_available(self):
        self.checked = True

    def execute(self, code, dataset, rows, execution_id):
        assert self.checked
        self.calls.append(code)
        status = next(self.statuses)
        return ExecutionRecord(
            execution_id,
            code,
            status,
            0.1,
            error="NameError: bad_variable" if status != "ok" else None,
            result=RESULT if status == "ok" else None,
            image="test-image",
        )


class FakePlanner:
    def __init__(self, sources):
        self.sources = iter(sources)
        self.contexts = []

    def propose(self, run):
        self.contexts.append(json.loads(json.dumps(asdict(run))))
        return PythonProposal("Check whether income relates to churn.", next(self.sources), "income", "churn")


def test_generated_loop_replans_using_failures_then_saves_executed_evidence():
    executor, planner = FakeExecutor(("error", "ok")), FakePlanner(["bad code", "fixed code"])
    run = investigate(
        demo_dataset(), "churn", budget=2, model="test", engine="python", executor=executor, planner=planner
    )
    assert [e.status for e in run.executions] == ["error", "ok"]
    assert len(run.evidence) == 1 and run.evidence[0].id == "A02"
    assert run.evidence[0].code == "fixed code"
    assert planner.contexts[1]["executions"][0]["error"] == "NameError: bad_variable"
    assert any(e["stage"] == "Replan" for e in run.activity)
    assert run.evidence[0].execution is run.executions[1]


def test_python_loop_does_not_repeat_successful_source():
    run = investigate(
        demo_dataset(),
        "churn",
        budget=2,
        model="test",
        engine="python",
        executor=FakeExecutor(),
        planner=FakePlanner(["same code", "same code"]),
    )
    assert len(run.evidence) == 1
    assert "repeated" in run.stop_reason


def test_missing_sandbox_never_generates_or_executes_code():
    class Unavailable(FakeExecutor):
        def check_available(self):
            raise SandboxUnavailable("missing")

    planner = FakePlanner([])
    with pytest.raises(SandboxUnavailable):
        investigate(demo_dataset(), "churn", model="test", engine="python", executor=Unavailable(), planner=planner)
    assert not planner.contexts
    with pytest.raises(ValueError, match="requires a local model"):
        investigate(demo_dataset(), "churn", engine="python")


def test_generated_exports_preserve_errors_and_replay_through_docker():
    run = investigate(
        demo_dataset(),
        "churn",
        budget=2,
        model="test",
        engine="python",
        executor=FakeExecutor(("error", "ok")),
        planner=FakePlanner(["bad code", "fixed code"]),
    )
    with ZipFile(BytesIO(export_bundle(run))) as archive:
        assert {"attempts/A01.json", "attempts/A02.json", "code/A02.py", "replay/A02.py"} <= set(archive.namelist())
        replay = archive.read("replay/A02.py").decode()
        assert "DockerSandbox().execute" in replay
        compile(replay, "<sandbox-replay>", "exec")
        assert "Failed execution attempts" in archive.read("report.md").decode()
        assert json.loads(archive.read("attempts/A01.json"))["status"] == "error"
    cells = json.loads(notebook(run))["cells"]
    code = "".join(cells[-1]["source"])
    assert "DockerSandbox().execute" in code
    assert "\nfixed code\n" not in code


def test_python_planner_validates_columns_and_source_and_passes_error_feedback():
    class Client:
        def generate_analysis(self, system, context):
            self.context = context
            return self.answer

    client = Client()
    client.answer = dict(hypothesis="Compare income and churn", x="income", y="churn", code="print(1)")
    planner = PythonPlanner("test", client)
    run = Investigation(profile_dataset(demo_dataset()), "churn", "", "test", 2)
    run.executions.append(ExecutionRecord("A01", "bad", "error", 0.1, error="bad variable", stderr="traceback"))
    assert planner.propose(run).code == "print(1)"
    assert client.context["attempts"][0]["error"] == "bad variable"
    assert "table" not in client.context
    client.answer["x"] = "invented column"
    with pytest.raises(PlannerError):
        planner.propose(run)
    client.answer["x"] = "income"
    client.answer["code"] = "x" * 70000
    with pytest.raises(PlannerError):
        planner.propose(run)
