"""Actual isolation tests. Explicit opt-in fails (never skips) if Docker is broken."""

import os
import subprocess
from tempfile import NamedTemporaryFile

import pytest

from pico.config import SandboxLimits
from pico.sandbox import DockerSandbox

pytestmark = pytest.mark.docker
DATASET = b"value,group\n1,a\n2,a\n3,b\n4,b\n"

RESULT_SOURCE = """
import json
from pathlib import Path
Path('/output/result.json').write_text(json.dumps({
    'n': 4, 'conclusion': 'Completed isolation fixture.',
    'limitations': ['Synthetic test data.'], 'table': [{'checked': True}],
    'statistic': None, 'p_value': None, 'effect': None
}))
"""


@pytest.fixture
def sandbox():
    if os.environ.get("PICO_RUN_DOCKER_TESTS") != "1":
        pytest.skip("Set PICO_RUN_DOCKER_TESTS=1 to run actual Docker isolation tests.")
    executor = DockerSandbox()
    executor.check_available()
    return executor


def containers():
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=pico-", "--format", "{{.Names}}"],
        capture_output=True,
        check=True,
        timeout=10,
    )
    return set(result.stdout.decode().splitlines())


def test_python_computes_real_evidence_and_returns_chart(sandbox):
    source = """
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
frame = pd.read_csv('/input/dataset.csv')
mean = float(frame['value'].mean())
frame.groupby('group')['value'].mean().plot.bar()
plt.title('Measured group means')
plt.tight_layout()
plt.savefig('/output/means.png')
print('Measured mean:', mean)
Path('/output/result.json').write_text(json.dumps({
    'n': len(frame), 'conclusion': f'Measured mean is {mean}.',
    'limitations': ['Four synthetic rows.'], 'table': [{'mean': mean}],
    'statistic': mean, 'p_value': None, 'effect': None
}))
"""
    before = containers()
    result = sandbox.execute(source, DATASET, 4)
    assert result.status == "ok", (result.error, result.stderr)
    assert result.result["statistic"] == 2.5
    assert "Measured mean: 2.5" in result.stdout
    assert result.artifacts[0].name == "means.png"
    assert containers() == before


def test_network_input_root_and_host_access_are_restricted(sandbox, monkeypatch):
    monkeypatch.setenv("PICO_HOST_SECRET", "must-not-reach-container")
    with NamedTemporaryFile(prefix="pico-host-secret-", delete=True) as sentinel:
        sentinel.write(b"host-only")
        sentinel.flush()
        source = (
            f"""
import os
import socket
from pathlib import Path
assert os.getuid() == 65532
assert 'PICO_HOST_SECRET' not in os.environ
assert not Path('/var/run/docker.sock').exists()
assert not Path({sentinel.name!r}).exists()
for path in ['/input/dataset.csv', '/input/analysis.py', '/etc/pico-escape']:
    try:
        Path(path).write_text('escape')
    except OSError:
        pass
    else:
        raise AssertionError('Unexpected writable path: ' + path)
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(('1.1.1.1', 80))
except OSError:
    pass
else:
    raise AssertionError('Network access was allowed')
finally:
    sock.close()
"""
            + RESULT_SOURCE
        )
        result = sandbox.execute(source, DATASET, 4)
    assert result.status == "ok", (result.error, result.stderr)


def test_runtime_errors_are_captured_and_not_findings(sandbox):
    result = sandbox.execute("raise RuntimeError('intentional failure')", DATASET, 4)
    assert result.status == "error" and result.result is None
    assert "RuntimeError: intentional failure" in result.stderr


def test_wall_timeout_removes_spawned_descendants(sandbox):
    executor = DockerSandbox(limits=SandboxLimits(wall_seconds=4))
    executor.check_available()
    before = containers()
    result = executor.execute(
        """
import subprocess, sys
subprocess.Popen([sys.executable, '-c', 'while True: pass'])
while True: pass
""",
        DATASET,
        4,
    )
    assert result.status == "timeout"
    assert result.duration_seconds < 20
    assert containers() == before


def test_memory_limit_does_not_crash_host(sandbox):
    executor = DockerSandbox(limits=SandboxLimits(memory_mb=256, wall_seconds=15))
    executor.check_available()
    result = executor.execute("blob = bytearray(2 * 1024 * 1024 * 1024)", DATASET, 4)
    assert result.status == "error" and result.result is None
    assert "exited with status" in result.error


def test_file_and_artifact_limits_are_enforced(sandbox):
    result = sandbox.execute(
        "open('/output/huge.png', 'wb').write(b'x' * (10 * 1024 * 1024))" + RESULT_SOURCE, DATASET, 4
    )
    assert result.status == "error" and result.result is None
    assert "Python exited with status" in result.error


def test_symlink_artifacts_are_rejected_without_host_file_copy(sandbox):
    source = "from pathlib import Path\nPath('/output/leak.png').symlink_to('/input/dataset.csv')\n" + RESULT_SOURCE
    result = sandbox.execute(source, DATASET, 4)
    assert result.status == "error" and result.artifacts == []
    assert "Invalid or oversized output" in result.error


def test_invalid_result_cannot_become_evidence(sandbox):
    source = RESULT_SOURCE + "\nPath('/output/result.json').write_text('{\"n\": 999999}')"
    result = sandbox.execute(source, DATASET, 4)
    assert result.status == "error" and result.result is None
    assert "Invalid sandbox output" in result.error


def test_generated_agent_repairs_real_execution_and_exports_provenance(sandbox):
    from io import BytesIO
    from zipfile import ZipFile

    from pico.agent import investigate
    from pico.data import load_csv
    from pico.planning import PythonPlanner
    from pico.report import export_bundle

    class RecordedModel:
        calls = 0

        def complete(self, system, context, schema):
            self.calls += 1
            if self.calls == 1:
                source = "raise NameError('repair this fixture')"
            else:
                assert context["attempts"][0]["status"] == "error"
                assert "repair this fixture" in context["attempts"][0]["stderr"]
                source = """
import json, pandas as pd
from pathlib import Path
frame = pd.read_csv('/input/dataset.csv')
mean = float(frame['value'].mean())
Path('/output/result.json').write_text(json.dumps({
    'n': len(frame), 'conclusion': f'Measured mean: {mean}',
    'limitations': ['Small synthetic sample.'], 'table': [{'mean': mean}],
    'statistic': mean, 'p_value': None, 'effect': None
}))
"""
            return dict(hypothesis="Measure mean value.", x="value", y="group", code=source)

    run = investigate(
        load_csv(DATASET),
        "group",
        budget=2,
        model="recorded-model",
        engine="python",
        executor=sandbox,
        planner=PythonPlanner("recorded-model", RecordedModel()),
    )
    assert [e.status for e in run.executions] == ["error", "ok"]
    assert len(run.evidence) == 1 and run.evidence[0].statistic == 2.5
    assert run.evidence[0].execution.image.startswith("sha256:")
    with ZipFile(BytesIO(export_bundle(run))) as bundle:
        assert "attempts/A01.json" in bundle.namelist()
        assert "DockerSandbox().execute" in bundle.read("replay/A02.py").decode()


def test_free_form_logistic_regression_example(sandbox):
    from pathlib import Path

    from pico.data import demo_dataset

    frame = demo_dataset()
    source = (Path(__file__).resolve().parents[1] / "examples" / "adjusted_churn.py").read_text()
    result = sandbox.execute(source, frame.to_csv(index=False).encode(), len(frame))
    assert result.status == "ok", (result.error, result.stderr)
    assert result.result["n"] == int(frame[["income", "purchases", "churn"]].notna().all(axis=1).sum())
    assert {row["term"] for row in result.result["table"]} == {"income", "purchases"}
    assert result.artifacts[0].name == "adjusted_churn.png"
