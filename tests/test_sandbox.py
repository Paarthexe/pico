"""Host-side protocol/transport tests never execute generated source on the host."""

import json
import subprocess
import sys
from base64 import b64encode
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from pico.config import MAX_PROTOCOL_BYTES, SandboxLimits
from pico.sandbox.docker import DockerSandbox, SandboxUnavailable, _capture, _ProcessOutput
from pico.sandbox.protocol import ProtocolError, decode_artifacts, decode_response, validate_result


@pytest.fixture
def valid_result():
    return dict(
        n=20,
        conclusion="The measured mean is 5.",
        limitations=["Synthetic fixture."],
        table=[{"mean": 5}],
        statistic=5,
        p_value=None,
        effect=None,
    )


def protocol(result):
    return json.dumps(
        dict(protocol_version=1, status="ok", result=result, stdout="finished", stderr="", artifacts=[])
    ).encode()


def test_protocol_requires_finite_evidence_and_valid_sample_size(valid_result):
    value, artifacts = decode_response(protocol(valid_result), 30)
    assert value["result"]["n"] == 20 and artifacts == []
    for update in [
        dict(n=True),
        dict(n=31),
        dict(p_value=1.1),
        dict(statistic=float("nan")),
        dict(table=[]),
        dict(table=[{"nested": [1, 2]}]),
        dict(limitations=[]),
    ]:
        with pytest.raises(ProtocolError):
            validate_result({**valid_result, **update}, 30)


def test_protocol_rejects_oversize_and_invalid_json(valid_result):
    with pytest.raises(ProtocolError):
        decode_response(b"x" * (MAX_PROTOCOL_BYTES + 1), 30)
    with pytest.raises(ProtocolError):
        decode_response(b"not json", 30)
    with pytest.raises(ProtocolError):
        decode_response(b"[]", 30)


def test_artifacts_validate_names_bytes_and_pixel_size():
    image = BytesIO()
    Image.new("RGB", (8, 8), "white").save(image, format="PNG")
    artifact = dict(name="chart.png", data_base64=b64encode(image.getvalue()).decode())
    assert decode_artifacts([artifact])[0].name == "chart.png"
    for bad in [
        {**artifact, "name": "../escape.png"},
        {**artifact, "data_base64": "not-base64"},
        {**artifact, "data_base64": b64encode(b"not a PNG").decode()},
    ]:
        with pytest.raises(ProtocolError):
            decode_artifacts([bad])
    with pytest.raises(ProtocolError):
        decode_artifacts([artifact, artifact])


def test_container_has_only_read_only_input_mount_and_enforced_limits():
    sandbox = DockerSandbox()
    sandbox._image_id = "sha256:" + "a" * 64
    command = sandbox.container_command("pico-test", Path("/tmp/inputs"))
    for flag in [
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--user=65532:65532",
        "--memory=768m",
        "--memory-swap=768m",
        "--cpus=1",
        "--pids-limit=64",
        "--log-driver=none",
        "--pull=never",
        "--init",
    ]:
        assert flag in command
    assert command.count("--mount") == 1
    assert "type=bind,src=/tmp/inputs,dst=/input,readonly" in command
    assert not any("docker.sock" in arg or "privileged" in arg for arg in command)
    assert command[-3:] == ["-I", "/opt/pico-worker.py", "30"]


def test_unavailable_docker_fails_closed(monkeypatch):
    monkeypatch.setattr("pico.sandbox.docker.shutil.which", lambda _: None)
    with pytest.raises(SandboxUnavailable, match="not installed"):
        DockerSandbox().execute("print(1)", b"x\n1\n", 1)


def test_cleanup_happens_on_timeout_and_malformed_output(monkeypatch, valid_result):
    calls = []
    sandbox = DockerSandbox()
    sandbox._image_id = "sha256:" + "a" * 64

    def control(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, b"container-id", b"")

    sandbox._control = control
    for captured, expected in [
        (_ProcessOutput(-9, b"", b"", "timeout"), "timeout"),
        (_ProcessOutput(0, b"bad JSON", b""), "error"),
        (_ProcessOutput(0, protocol(valid_result), b""), "ok"),
    ]:
        monkeypatch.setattr("pico.sandbox.docker._capture", lambda *args: captured)
        record = sandbox.execute("print(1)", b"x\n1\n", 30)
        assert record.status == expected
        assert calls[-1][:2] == ("rm", "--force")
        assert calls[-1][-1].startswith("pico-")
    assert record.result["n"] == 20


def test_capture_is_bounded_and_has_wall_timeout():
    # These are fixed transport fixtures, not user or model-generated source.
    output = _capture([sys.executable, "-c", "import time; time.sleep(5)"], 0.1)
    assert output.status == "timeout"
    output = _capture([sys.executable, "-c", 'import sys; sys.stdout.write("x" * 8000000)'], 5)
    assert output.status == "output_limit"
    assert len(output.stdout) <= MAX_PROTOCOL_BYTES


def test_limits_cannot_be_disabled():
    for kwargs in [dict(wall_seconds=0), dict(wall_seconds=float("inf")), dict(memory_mb=0), dict(pids=True)]:
        with pytest.raises(ValueError):
            SandboxLimits(**kwargs)
