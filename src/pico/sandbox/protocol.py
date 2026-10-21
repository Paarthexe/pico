"""Validate every byte crossing back from an untrusted analysis container."""

import base64
import json
import math
import re
import warnings
from io import BytesIO

from PIL import Image

from pico.config import (
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_TOTAL_BYTES,
    MAX_ARTIFACTS,
    MAX_LOG_BYTES,
    MAX_PROTOCOL_BYTES,
    MAX_RESULT_BYTES,
    PROTOCOL_VERSION,
)
from pico.models import Artifact


class ProtocolError(ValueError):
    pass


def _text(value, label: str, maximum: int) -> str:
    try:
        valid = isinstance(value, str) and len(value.encode()) <= maximum
    except UnicodeError:
        valid = False
    if not valid:
        raise ProtocolError(f"Invalid {label}.")
    return value


def _finite(value: object) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def validate_result(value: object, dataset_rows: int) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError("result.json must contain a JSON object.")
    try:
        size = len(json.dumps(value, allow_nan=False).encode())
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise ProtocolError("Result contains non-JSON values.") from exc
    if size > MAX_RESULT_BYTES:
        raise ProtocolError("Result exceeded 256 KB.")
    n = value.get("n")
    if type(n) is not int or not 1 <= n <= dataset_rows:
        raise ProtocolError("Result n must be a positive integer no greater than dataset rows.")
    conclusion = _text(value.get("conclusion"), "conclusion", 8000)
    if not conclusion.strip():
        raise ProtocolError("Result conclusion is empty.")
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not 1 <= len(limitations) <= 20:
        raise ProtocolError("Result must include 1–20 limitations.")
    for item in limitations:
        _text(item, "limitation", 2000)
    table = value.get("table")
    if not isinstance(table, list) or not 1 <= len(table) <= 1000:
        raise ProtocolError("Result must include a nonempty evidence table with at most 1000 rows.")
    for row in table:
        if not isinstance(row, dict) or not 1 <= len(row) <= 20:
            raise ProtocolError("Evidence rows must contain 1–20 scalar fields.")
        for key, cell in row.items():
            _text(key, "table column", 200)
            if cell is None or type(cell) is bool:
                continue
            if isinstance(cell, str):
                _text(cell, "table cell", 2000)
            elif not _finite(cell):
                raise ProtocolError("Evidence table values must be finite JSON scalars.")
    result = dict(n=n, conclusion=conclusion, limitations=limitations, table=table)
    for field in ["statistic", "p_value", "effect"]:
        number = value.get(field)
        if number is not None:
            if not _finite(number):
                raise ProtocolError(f"Invalid {field}: expected a finite number or null.")
            if field == "p_value" and not 0 <= number <= 1:
                raise ProtocolError("p_value must be between 0 and 1.")
        result[field] = number
    return result


def decode_artifacts(values: object) -> list[Artifact]:
    if not isinstance(values, list) or len(values) > MAX_ARTIFACTS:
        raise ProtocolError("Too many chart artifacts.")
    artifacts, names, total = [], set(), 0
    for value in values:
        if not isinstance(value, dict):
            raise ProtocolError("Malformed chart artifact.")
        name = value.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.png", name):
            raise ProtocolError("Chart filename must be a plain PNG filename.")
        if name in names:
            raise ProtocolError("Duplicate chart filename.")
        names.add(name)
        encoded = value.get("data_base64")
        if not isinstance(encoded, str) or len(encoded) > (MAX_ARTIFACT_BYTES * 4 // 3 + 8):
            raise ProtocolError("Chart exceeds the 2 MB limit.")
        try:
            data = base64.b64decode(encoded, validate=True)
            total += len(data)
            if len(data) > MAX_ARTIFACT_BYTES or total > MAX_ARTIFACT_TOTAL_BYTES:
                raise ProtocolError("Charts exceed the output size budget.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as picture:
                    if picture.format != "PNG" or picture.width * picture.height > 8_000_000:
                        raise ProtocolError("Chart must be PNG and at most 8 million pixels.")
                    picture.verify()
        except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            if isinstance(exc, ProtocolError):
                raise
            raise ProtocolError("Invalid or unsafe PNG chart.") from exc
        artifacts.append(Artifact(name, encoded))
    return artifacts


def decode_response(raw: bytes, dataset_rows: int) -> tuple[dict, list[Artifact]]:
    if len(raw) > MAX_PROTOCOL_BYTES:
        raise ProtocolError("Sandbox response exceeded its size limit.")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("Sandbox did not return valid JSON.") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("protocol_version")) is not int
        or value.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ProtocolError("Unsupported sandbox protocol.")
    if not isinstance(value.get("status"), str) or value.get("status") not in {"ok", "error", "timeout"}:
        raise ProtocolError("Invalid sandbox status.")
    for field in ["stdout", "stderr"]:
        _text(value.get(field, ""), field, MAX_LOG_BYTES)
    error = value.get("error")
    if error is not None:
        _text(error, "sandbox error", 8000)
    if value["status"] != "ok":
        return value, []
    value["result"] = validate_result(value.get("result"), dataset_rows)
    return value, decode_artifacts(value.get("artifacts", []))
