"""Generated-analysis loop: propose → isolated execution → validate → evaluate/replan."""

import hashlib
from collections.abc import Callable
from dataclasses import asdict

import pandas as pd

from pico.config import EXPLORATORY_CAVEAT, MULTIPLE_TESTING_CAVEAT
from pico.models import Evidence, ExecutionRecord, Hypothesis, Investigation, PythonProposal
from pico.ollama import PlannerError
from pico.planning import PythonPlanner
from pico.ports import ProposalPlanner, PythonExecutor
from pico.review import review_investigation
from pico.sandbox import DockerSandbox
from pico.sandbox.protocol import validate_result


def evidence_from_execution(record: ExecutionRecord, proposal: PythonProposal, dataset_rows: int) -> Evidence:
    if record.status != "ok" or record.result is None:
        raise ValueError("Only a successful, validated execution can become a finding.")
    result = validate_result(record.result, dataset_rows)
    limitations = [
        EXPLORATORY_CAVEAT,
        MULTIPLE_TESTING_CAVEAT,
        *result["limitations"],
        "Generated analysis code and its statistical interpretation require domain review.",
    ]
    if result["n"] < 50:
        limitations.append("Small sample: fewer than 50 usable observations.")
    h = Hypothesis("python", proposal.x, proposal.y, proposal.hypothesis)
    return Evidence(
        record.id,
        asdict(h),
        result["n"],
        result["statistic"],
        result["p_value"],
        result["effect"],
        result["conclusion"],
        list(dict.fromkeys(limitations)),
        result["table"],
        proposal.code,
        execution=record,
        method=result.get("method", ""),
        assumptions=result.get("assumptions", []),
        diagnostics=result.get("diagnostics", {}),
    )


def python_loop(
    df: pd.DataFrame,
    run: Investigation,
    emit: Callable[[str, str], None],
    *,
    executor: PythonExecutor | None = None,
    planner: ProposalPlanner | None = None,
) -> None:
    executor = executor or DockerSandbox()
    planner = planner or PythonPlanner(run.model)
    # Fail before model generation when the security boundary is unavailable.
    executor.check_available()
    dataset = df.to_csv(index=False).encode()
    successful_sources = set()
    emit("Plan", f"Generate and execute up to {run.budget} Python investigations in disposable containers.")
    for index in range(1, run.budget + 1):
        try:
            proposal = planner.propose(run)
        except PlannerError as exc:
            emit("Planner error", str(exc))
            run.stop_reason = "The model could not produce a valid Python investigation."
            return
        digest = hashlib.sha256(proposal.code.encode()).hexdigest()
        if digest in successful_sources:
            emit(
                "Plan revision",
                "The proposal repeats a completed analysis and was not executed. Choose a new relationship or a deeper control-variable comparison, not the same source with cosmetic edits.",
            )
            continue
        emit("Investigate", proposal.hypothesis)
        record = executor.execute(proposal.code, dataset, len(df), f"A{index:02}")
        run.executions.append(record)
        if record.status != "ok":
            emit("Execution failed", f"{record.id}: {record.error or record.status}")
            if index < run.budget:
                emit("Replan", "Return the execution error to the model for a repaired or different analysis.")
            continue
        evidence = evidence_from_execution(record, proposal, len(df))
        run.evidence.append(evidence)
        review_investigation(df, run)
        successful_sources.add(digest)
        emit("Evaluate", f"{evidence.id}: {evidence.conclusion} n={evidence.n}.")
        if index < run.budget:
            emit("Replan", "Use the executed results to decide the next investigation.")
    run.stop_reason = f"Reached the {run.budget}-analysis budget."
