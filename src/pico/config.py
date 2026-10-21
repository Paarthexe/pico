"""Application limits and the versioned sandbox protocol contract."""

import math
from dataclasses import dataclass

MAX_INVESTIGATIONS = 5
MAX_CODE_BYTES = 64 * 1024
MAX_RESULT_BYTES = 256 * 1024
MAX_LOG_BYTES = 64 * 1024
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_TOTAL_BYTES = 4 * 1024 * 1024
MAX_ARTIFACTS = 4
MAX_PROTOCOL_BYTES = 7 * 1024 * 1024
SANDBOX_IMAGE = "pico-sandbox:1"
PROTOCOL_VERSION = 1
EXPLORATORY_CAVEAT = "Exploratory association; this does not establish causation."
MULTIPLE_TESTING_CAVEAT = "P-values are unadjusted; account for the full investigation when interpreting them."


@dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: float = 45
    cpu_seconds: int = 30
    memory_mb: int = 768
    pids: int = 64
    output_mb: int = 32
    temporary_mb: int = 64

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_seconds) or not 1 <= self.wall_seconds <= 300:
            raise ValueError("Sandbox wall timeout must be between 1 and 300 seconds.")
        for name, minimum, maximum in [
            ("cpu_seconds", 1, 300),
            ("memory_mb", 256, 2048),
            ("pids", 16, 128),
            ("output_mb", 8, 64),
            ("temporary_mb", 16, 128),
        ]:
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Invalid sandbox limit: {name}.")
