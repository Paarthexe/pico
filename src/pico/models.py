"""Shared domain records; independent of UI, model providers, and execution engines."""

from dataclasses import dataclass, field
from typing import Literal

AnalysisKind = Literal["correlation", "groups", "numeric_groups", "missingness", "stratified", "python"]
ExecutionStatus = Literal["ok", "error", "timeout", "output_limit", "unavailable"]


@dataclass(frozen=True)
class Hypothesis:
    kind: AnalysisKind
    x: str
    y: str
    reason: str
    control: str | None = None

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        return self.kind, self.x, self.y, self.control


@dataclass(frozen=True)
class Artifact:
    """Validated PNG stored inline so a completed run needs no container or temp files."""

    name: str
    data_base64: str
    media_type: str = "image/png"


@dataclass
class ExecutionRecord:
    id: str
    code: str
    status: ExecutionStatus
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    result: dict | None = None
    image: str | None = None


@dataclass
class FindingReview:
    original_conclusion: str
    status: str = "caution"
    adjusted_p_value: float | None = None
    method: str = "Holm"
    family_size: int = 0
    adjusted_tests: list[dict] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    numerical_summary: str = ""


@dataclass
class Evidence:
    id: str
    hypothesis: dict
    n: int
    statistic: float | None
    p_value: float | None
    effect: float | None
    conclusion: str
    limitations: list[str]
    table: list[dict]
    code: str
    error: str | None = None
    execution: ExecutionRecord | None = None
    method: str = ""
    assumptions: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    p_value_reliable: bool = True
    review: FindingReview | None = None


@dataclass(frozen=True)
class PythonProposal:
    hypothesis: str
    code: str
    x: str
    y: str


@dataclass(frozen=True)
class StudyContext:
    subject_id: str | None = None
    observation_time: str | None = None
    outcome_time: str | None = None
    post_outcome_columns: tuple[str, ...] = ()

    @property
    def excluded_columns(self) -> set[str]:
        return {c for c in [self.subject_id, self.observation_time, self.outcome_time, *self.post_outcome_columns] if c}


@dataclass
class Investigation:
    profile: dict
    target: str
    question: str
    mode: str
    budget: int
    evidence: list[Evidence] = field(default_factory=list)
    activity: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    executions: list[ExecutionRecord] = field(default_factory=list)
    engine: str = "catalog"
    model: str | None = None
    dataset_warnings: list[dict] = field(default_factory=list)
    multiplicity: dict = field(default_factory=dict)
    study_context: StudyContext = field(default_factory=StudyContext)
