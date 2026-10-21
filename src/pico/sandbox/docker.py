"""Docker executor: no generated source is ever executed in the application process."""

import json
import os
import re
import selectors
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from pico.config import MAX_CODE_BYTES, MAX_LOG_BYTES, MAX_PROTOCOL_BYTES, SANDBOX_IMAGE, SandboxLimits
from pico.data import MAX_BYTES, MAX_ROWS
from pico.models import ExecutionRecord
from pico.sandbox.protocol import ProtocolError, decode_response


class SandboxUnavailable(RuntimeError):
    """The isolation boundary cannot be established; execution must not fall back to host."""


@dataclass(frozen=True)
class _ProcessOutput:
    returncode: int
    stdout: bytes
    stderr: bytes
    status: str = "ok"


def _capture(command: list[str], timeout: float) -> _ProcessOutput:
    """Bounded streaming capture, including code that bypasses worker log capture."""
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = {"stdout": bytearray(), "stderr": bytearray()}
    deadline, status = time.monotonic() + timeout, "ok"
    try:
        with selectors.DefaultSelector() as selector:
            for name, stream in [("stdout", process.stdout), ("stderr", process.stderr)]:
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    name = key.data
                    maximum = MAX_PROTOCOL_BYTES if name == "stdout" else MAX_LOG_BYTES
                    if len(output[name]) + len(chunk) > maximum:
                        status = "output_limit"
                        break
                    output[name].extend(chunk)
                if status != "ok":
                    break
            if status == "ok":
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    status = "timeout"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
    return _ProcessOutput(process.returncode, bytes(output["stdout"]), bytes(output["stderr"]), status)


class DockerSandbox:
    def __init__(self, image: str = SANDBOX_IMAGE, limits: SandboxLimits | None = None, binary: str = "docker"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,255}", image):
            raise ValueError("Invalid sandbox image name.")
        self.image = image
        self.limits = limits or SandboxLimits()
        self.binary = binary
        self._image_id: str | None = None

    def _control(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.binary, *arguments], stdin=subprocess.DEVNULL, capture_output=True, timeout=10, check=False
        )

    def check_available(self) -> None:
        if shutil.which(self.binary) is None:
            raise SandboxUnavailable(
                "Docker is not installed. Install and start Docker, then build the Pico sandbox image."
            )
        try:
            info = self._control("info", "--format", "{{json .}}")
            if info.returncode:
                raise SandboxUnavailable("Docker daemon is unavailable. Start Docker and try again.")
            details = json.loads(info.stdout)
            if details.get("OSType") != "linux":
                raise SandboxUnavailable("Pico requires Linux containers.")
            if not any("name=seccomp,profile=builtin" == item for item in details.get("SecurityOptions", [])):
                raise SandboxUnavailable("Docker's built-in seccomp profile is required for generated Python.")
            if not all(
                details.get(capability) is True
                for capability in ["MemoryLimit", "SwapLimit", "PidsLimit", "CpuCfsQuota"]
            ):
                raise SandboxUnavailable("Docker must support memory, swap, process, and CPU quota limits.")
            image = self._control("image", "inspect", "--format", "{{.Id}}", self.image)
            identifier = image.stdout.decode().strip()
            if image.returncode or not re.fullmatch(r"sha256:[a-f0-9]{64}", identifier):
                raise SandboxUnavailable(
                    "Sandbox image is missing. Run: docker build -f sandbox/Dockerfile -t pico-sandbox:1 ."
                )
            self._image_id = identifier
        except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError, TypeError) as exc:
            raise SandboxUnavailable(f"Could not verify Docker isolation ({type(exc).__name__}).") from exc

    def container_command(self, name: str, input_directory: Path) -> list[str]:
        """Only read-only inputs are mounted; all writable space is size-limited tmpfs."""
        if self._image_id is None:
            raise SandboxUnavailable("Verify the sandbox before creating a container.")
        if "," in str(input_directory):
            raise SandboxUnavailable("Docker input paths cannot contain commas.")
        limits = self.limits
        return [
            self.binary,
            "create",
            "--name",
            name,
            "--pull=never",
            "--init",
            "--network=none",
            "--ipc=none",
            "--cgroupns=private",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges=true",
            "--user=65532:65532",
            f"--memory={limits.memory_mb}m",
            f"--memory-swap={limits.memory_mb}m",
            "--cpus=1",
            f"--pids-limit={limits.pids}",
            "--ulimit=nofile=128:128",
            "--ulimit=fsize=8388608:8388608",
            "--ulimit=core=0:0",
            "--log-driver=none",
            "--mount",
            f"type=bind,src={input_directory / 'dataset.csv'},dst=/input/dataset.csv,readonly",
            "--mount",
            f"type=bind,src={input_directory / 'analysis.py'},dst=/input/analysis.py,readonly",
            "--tmpfs",
            f"/output:rw,noexec,nosuid,nodev,size={limits.output_mb}m,uid=65532,gid=65532,mode=700",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={limits.temporary_mb}m,uid=65532,gid=65532,mode=700",
            "--env=OPENBLAS_NUM_THREADS=1",
            "--env=OMP_NUM_THREADS=1",
            "--env=MKL_NUM_THREADS=1",
            "--env=NUMEXPR_NUM_THREADS=1",
            "--env=MPLBACKEND=Agg",
            "--env=MPLCONFIGDIR=/tmp/matplotlib",
            "--env=HOME=/tmp",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=PYTHONUNBUFFERED=1",
            "--workdir=/output",
            "--entrypoint=python",
            self._image_id,
            "-I",
            "/opt/pico-worker.py",
            str(limits.cpu_seconds),
        ]

    def execute(self, code: str, dataset: bytes, dataset_rows: int, execution_id: str = "A01") -> ExecutionRecord:
        if not isinstance(code, str) or not code.strip() or len(code.encode()) > MAX_CODE_BYTES:
            raise ValueError("Python source must be nonempty and no larger than 64 KB.")
        if not isinstance(dataset, bytes) or not 0 < len(dataset) <= MAX_BYTES:
            raise ValueError("Sandbox dataset must be nonempty and no larger than 20 MB.")
        if type(dataset_rows) is not int or not 1 <= dataset_rows <= MAX_ROWS:
            raise ValueError("Invalid sandbox dataset row count.")
        if self._image_id is None:
            self.check_available()
        started, name = time.monotonic(), f"pico-{uuid4().hex}"
        record = ExecutionRecord(execution_id, code, "error", 0, image=self._image_id)
        # No artifact files are copied to the host. Only bounded JSON/PNG bytes return.
        with TemporaryDirectory(prefix="pico-input-") as directory:
            # macOS exposes /var as /private/var, while Linux VM file shares
            # use the canonical path. Bind the path actually shared with Docker.
            inputs = Path(directory).resolve()
            # The parent stays private on the host. Docker mounts individual files,
            # so its unprivileged child can read them without exposing the directory
            # to other local users.
            inputs.chmod(0o700)
            for filename, contents in [("dataset.csv", dataset), ("analysis.py", code.encode())]:
                path = inputs / filename
                path.write_bytes(contents)
                path.chmod(0o444)
            try:
                created = self._control(*self.container_command(name, inputs)[1:])
                if created.returncode:
                    record.error = "Docker refused the restricted container: " + created.stderr[:2000].decode(
                        errors="replace"
                    )
                else:
                    remaining = self.limits.wall_seconds - (time.monotonic() - started)
                    output = _capture([self.binary, "start", "--attach", name], max(0.001, remaining))
                    self._interpret(output, dataset_rows, record)
            except (OSError, subprocess.TimeoutExpired) as exc:
                record.error = f"Docker execution failed ({type(exc).__name__})."
            finally:
                # This kills the entire container, including spawned descendants, even
                # on timeout, malformed output, keyboard interruption, or client error.
                try:
                    removed = self._control("rm", "--force", name)
                    if removed.returncode and b"No such container" not in removed.stderr:
                        raise SandboxUnavailable("Docker could not remove the analysis container; check the daemon.")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise SandboxUnavailable(
                        "Container cleanup could not be confirmed; check Docker before retrying."
                    ) from exc
        record.duration_seconds = round(time.monotonic() - started, 3)
        return record

    @staticmethod
    def _interpret(output: _ProcessOutput, dataset_rows: int, record: ExecutionRecord) -> None:
        if output.status != "ok":
            record.status = output.status
            record.error = (
                "Sandbox exceeded its wall-time limit."
                if output.status == "timeout"
                else "Sandbox exceeded its response output limit."
            )
            return
        if output.returncode:
            record.error = f"Container exited with status {output.returncode} (resource limit or runtime failure)."
            record.stderr = output.stderr[:MAX_LOG_BYTES].decode(errors="replace")
            return
        try:
            response, artifacts = decode_response(output.stdout, dataset_rows)
        except (ProtocolError, RecursionError, OverflowError) as exc:
            record.error = f"Invalid sandbox output: {exc}"
            return
        record.status = response["status"]
        record.stdout = response.get("stdout", "")
        record.stderr = response.get("stderr", "")
        record.error = response.get("error")
        record.result = response.get("result") if record.status == "ok" else None
        record.artifacts = artifacts
