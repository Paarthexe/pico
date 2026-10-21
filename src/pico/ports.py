"""Structural interfaces used by orchestration and injectable test doubles."""

from typing import Protocol

from pico.models import ExecutionRecord, Investigation, PythonProposal


class ModelClient(Protocol):
    def complete(self, system: str, context: dict, schema: dict | None = None) -> dict: ...


class ProposalPlanner(Protocol):
    def propose(self, run: Investigation) -> PythonProposal: ...


class PythonExecutor(Protocol):
    def check_available(self) -> None: ...

    def execute(self, code: str, dataset: bytes, dataset_rows: int, execution_id: str = "A01") -> ExecutionRecord: ...
