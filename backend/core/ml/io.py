"""Robust CSV input/output helpers and JSON sanitisation.

Every value that leaves the ML layer for an HTTP response passes through
:func:`json_safe`, which turns numpy/pandas scalars into builtins and maps
NaN/Inf to ``None`` so the payload is always valid JSON.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import numbers
import os
import re
import uuid
from decimal import Decimal
from enum import Enum
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024
MAX_PREVIEW_ROWS: int = 200

#: Delimiters we are willing to detect automatically.
CANDIDATE_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")

#: Encodings tried in order when decoding an uploaded file.
ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")

_SNIFF_BYTES = 64 * 1024
_MAX_DEPTH = 60
_UNNAMED_RE = re.compile(r"^Unnamed:\s*\d+(_level_\d+)?$")
_DELIMITER_ALIASES = {
    "auto": "auto",
    "": "auto",
    "tab": "\t",
    "\\t": "\t",
    "comma": ",",
    "semicolon": ";",
    "pipe": "|",
}


class DataError(Exception):
    """Raised for any user-facing data problem. ``str(e)`` is shown to the user."""


# --------------------------------------------------------------------------- #
# Delimiter detection
# --------------------------------------------------------------------------- #
def sniff_delimiter(sample: str) -> str:
    """Guess the column separator used by ``sample``.

    Uses :class:`csv.Sniffer` first and falls back to a per-line frequency
    count. Returns ``","`` when nothing conclusive is found.
    """
    if not sample or not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS))
        if dialect.delimiter in CANDIDATE_DELIMITERS:
            return dialect.delimiter
    except (csv.Error, TypeError, ValueError):
        pass

    lines = [line for line in sample.splitlines() if line.strip()][:10]
    if len(lines) > 1:
        # The last line of a truncated sample may be incomplete.
        lines = lines[:-1] if len(lines) > 2 else lines

    best_delimiter = ","
    best_score = 0
    for candidate in CANDIDATE_DELIMITERS:
        counts = [line.count(candidate) for line in lines]
        if not counts or counts[0] == 0:
            continue
        if any(count != counts[0] for count in counts):
            continue
        if counts[0] > best_score:
            best_delimiter = candidate
            best_score = counts[0]
    return best_delimiter if best_score else ","


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
#: Leading bytes that identify a file we can never read as a table.
_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "a zip archive (an .xlsx spreadsheet, for example)"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"%PDF", "a PDF document"),
    (b"\x89PNG", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF8", "a GIF image"),
    (b"\xd0\xcf\x11\xe0", "an old Office document"),
    (b"SQLite format 3\x00", "a SQLite database"),
    (b"\x7fELF", "a program"),
    (b"{\\rtf", "an RTF document"),
)

#: Control bytes that never occur in a real delimited text file.
_CONTROL_BYTES = frozenset(
    set(range(0x00, 0x09)) | {0x0B, 0x0C} | set(range(0x0E, 0x20)) | {0x7F}
)

#: Above this share of control bytes the file is treated as binary.
_MAX_CONTROL_RATIO = 0.05


def _reject_binary(raw: bytes) -> None:
    """Raise :class:`DataError` when the bytes are clearly not delimited text."""
    head = raw[:_SNIFF_BYTES]
    for signature, description in _BINARY_SIGNATURES:
        if head.startswith(signature):
            raise DataError(
                f"That file looks like {description}, not a CSV. Save your data as a "
                "CSV file and upload that instead."
            )
    if b"\x00" in head:
        raise DataError(
            "That file is not a text file, so it cannot be read as a table. Save your "
            "data as a CSV file and upload that instead."
        )
    if head:
        control = sum(1 for byte in head if byte in _CONTROL_BYTES)
        if control / len(head) > _MAX_CONTROL_RATIO:
            raise DataError(
                "That file is not a text file, so it cannot be read as a table. Save your "
                "data as a CSV file and upload that instead."
            )


def _too_large() -> DataError:
    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return DataError(
        f"The file is larger than the {limit_mb} MB upload limit. "
        "Upload a smaller file or split it into parts."
    )


def _rewind(handle: Any) -> None:
    """Seek a file-like object back to the start, ignoring any failure."""
    seek = getattr(handle, "seek", None)
    if callable(seek):
        try:
            seek(0)
        except (OSError, ValueError):
            pass


def _reopen(handle: Any) -> Any | None:
    """Best-effort reopen of a Django ``FieldFile``. Returns None when impossible."""
    opener = getattr(handle, "open", None)
    if not callable(opener):
        return None
    for args in (("rb",), ()):
        try:
            reopened = opener(*args)
        except Exception:
            continue
        candidate = reopened if hasattr(reopened, "read") else handle
        _rewind(candidate)
        return candidate
    return None


def _read_raw_bytes(path_or_file: Any) -> bytes:
    """Return the full contents of ``path_or_file`` as bytes, size-capped."""
    if isinstance(path_or_file, (bytes, bytearray, memoryview)):
        raw = bytes(path_or_file)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise _too_large()
        return raw

    if isinstance(path_or_file, (str, os.PathLike)):
        path = os.fspath(path_or_file)
        if not os.path.exists(path):
            raise DataError("The data file could not be found on the server.")
        try:
            if os.path.getsize(path) > MAX_UPLOAD_BYTES:
                raise _too_large()
            with open(path, "rb") as handle:
                return handle.read(MAX_UPLOAD_BYTES + 1)
        except OSError as exc:
            raise DataError(f"The data file could not be read: {exc}") from exc

    if not hasattr(path_or_file, "read"):
        raise DataError("The uploaded file could not be read.")

    declared_size = getattr(path_or_file, "size", None)
    if isinstance(declared_size, numbers.Integral) and int(declared_size) > MAX_UPLOAD_BYTES:
        raise _too_large()

    handle = path_or_file
    if getattr(handle, "closed", False):
        handle = _reopen(handle) or handle
    _rewind(handle)

    try:
        chunk = handle.read(MAX_UPLOAD_BYTES + 1)
    except (OSError, ValueError) as exc:
        # A Django FieldFile that was closed underneath us: reopen and retry once.
        retry = _reopen(path_or_file)
        if retry is None:
            raise DataError(f"The uploaded file could not be read: {exc}") from exc
        handle = retry
        try:
            chunk = handle.read(MAX_UPLOAD_BYTES + 1)
        except (OSError, ValueError) as retry_exc:
            raise DataError(f"The uploaded file could not be read: {retry_exc}") from retry_exc

    # Leave the handle usable for the caller (e.g. saving the same upload).
    _rewind(handle)

    if chunk is None:
        chunk = b""
    if isinstance(chunk, str):
        chunk = chunk.encode("utf-8", errors="replace")
    if len(chunk) > MAX_UPLOAD_BYTES:
        raise _too_large()
    return bytes(chunk)


def _resolve_delimiter(raw: bytes, delimiter: str) -> tuple[str, bool]:
    """Return ``(separator, was_auto_detected)``."""
    requested = delimiter if isinstance(delimiter, str) else "auto"
    requested = _DELIMITER_ALIASES.get(requested.strip().lower(), requested)
    if requested == "auto":
        sample = raw[:_SNIFF_BYTES].decode("latin-1", errors="replace")
        return sniff_delimiter(sample), True
    if not requested:
        return ",", False
    return requested, False


def _clean_column_names(columns: Any) -> list[str]:
    """Strip, de-blank and de-duplicate raw header names."""
    cleaned: list[str] = []
    for index, raw_name in enumerate(columns):
        name = str(raw_name).replace("﻿", "").strip()
        if not name or name.lower() == "nan" or _UNNAMED_RE.match(name):
            name = f"column_{index + 1}"
        cleaned.append(name)

    seen: dict[str, int] = {}
    result: list[str] = []
    for name in cleaned:
        if name not in seen:
            seen[name] = 0
            result.append(name)
            continue
        seen[name] += 1
        candidate = f"{name}.{seen[name]}"
        while candidate in seen:
            seen[name] += 1
            candidate = f"{name}.{seen[name]}"
        seen[candidate] = 0
        result.append(candidate)
    return result


def read_csv(path_or_file: Any, delimiter: str = "auto") -> pd.DataFrame:
    """Read a delimited text file into a DataFrame.

    Accepts a filesystem path, a file-like object or a Django ``UploadedFile``.
    Tries the utf-8, utf-8-sig and latin-1 encodings in order. Header names are
    stripped, de-duplicated (``a``, ``a.1``) and blank ones become ``column_N``.
    Raises :class:`DataError` with a human-readable message for anything the
    user needs to fix.
    """
    raw = _read_raw_bytes(path_or_file)
    if not raw.strip():
        raise DataError(
            "The file is empty. Upload a CSV file with a header row and at least one data row."
        )
    _reject_binary(raw)

    separator, auto_detected = _resolve_delimiter(raw, delimiter)

    df: pd.DataFrame | None = None
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        buffer = BytesIO(raw)
        buffer.seek(0)
        try:
            df = pd.read_csv(
                buffer,
                sep=separator,
                encoding=encoding,
                engine="python" if len(separator) > 1 else "c",
                index_col=None,
                skip_blank_lines=True,
            )
            break
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataError(
                "The file does not contain any columns. Check that it is a CSV file "
                "with a header row."
            ) from exc
        except pd.errors.ParserError as exc:
            raise DataError(
                "The file could not be parsed as a table. Check that every row has the "
                "same number of columns, then try again with an explicit delimiter."
            ) from exc
        except ValueError as exc:
            raise DataError(f"The file could not be read as a table: {exc}") from exc

    if df is None:
        raise DataError(
            "The file's text could not be decoded. Save it as UTF-8 CSV and upload it again."
        ) from last_error

    if df.shape[1] == 0:
        raise DataError(
            "The file does not contain any columns. Check that it is a CSV file with a header row."
        )

    if df.shape[1] == 1:
        # A real separator the parser missed always shows up in the header row;
        # quoted data cells may legitimately contain one, so only judge the header.
        header = str(df.columns[0])
        if any(candidate in header for candidate in CANDIDATE_DELIMITERS):
            if auto_detected:
                raise DataError(
                    "The column separator could not be detected, so every row was read as a "
                    "single column. Upload the file again and choose the delimiter explicitly."
                )
            raise DataError(
                "Every row was read as a single column, so the file does not use the "
                "delimiter you chose. Pick the delimiter that matches the file."
            )

    if len(df) == 0:
        raise DataError(
            "The file has a header row but no data rows. Add some rows and upload it again."
        )

    df.columns = pd.Index(_clean_column_names(df.columns))
    return df


# --------------------------------------------------------------------------- #
# JSON sanitisation
# --------------------------------------------------------------------------- #
def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _json_safe(value: Any, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return str(value)

    if value is None or value is pd.NaT or value is pd.NA:
        return None

    # Containers first: pd.isna() on a container returns an array.
    if isinstance(value, dict):
        return {str(key): _guard(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_guard(item, depth + 1) for item in value]
    if isinstance(value, np.ndarray):
        return [_guard(item, depth + 1) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_guard(record, depth + 1) for record in value.to_dict(orient="records")]
    if isinstance(value, (pd.Series, pd.Index, pd.Categorical)):
        return [_guard(item, depth + 1) for item in value.tolist()]

    # Scalars.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, (int, np.integer)) or isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(float(value))
    if isinstance(value, Decimal):
        return _finite_or_none(float(value))
    if isinstance(value, complex) or isinstance(value, np.complexfloating):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if value is pd.NaT else value.isoformat()
    if isinstance(value, np.datetime64):
        stamp = pd.Timestamp(value)
        return None if stamp is pd.NaT else stamp.isoformat()
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, dt.timedelta, np.timedelta64)):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Enum):
        return _guard(value.value, depth + 1)

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if missing is True:
        return None

    if isinstance(value, np.generic):
        return _guard(value.item(), depth + 1)

    return str(value)


def _guard(value: Any, depth: int) -> Any:
    try:
        return _json_safe(value, depth)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def json_safe(value: Any) -> Any:
    """Recursively convert ``value`` into JSON-serialisable builtins.

    numpy/pandas scalars become Python builtins, containers are converted
    element-wise, NaN/Inf/-Inf and pandas NA/NaT become ``None``. Never raises.
    """
    return _guard(value, 0)


# --------------------------------------------------------------------------- #
# Preview / writing
# --------------------------------------------------------------------------- #
def _preview_cell(value: Any) -> Any:
    """One preview cell, kept inside the contract's ``str|float|int|null`` union.

    Booleans are shown the way they are written to the stored CSV so a client
    that renders the union does not have to handle a fourth scalar type.
    """
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return json_safe(value)


def dataframe_preview(df: pd.DataFrame, n_rows: int = 10) -> dict:
    """Return ``{"headers": [...], "rows": [[...]]}`` for the first rows."""
    try:
        limit = int(n_rows)
    except (TypeError, ValueError):
        limit = 10
    limit = max(0, min(limit, MAX_PREVIEW_ROWS))

    headers = [str(column) for column in df.columns]
    rows: list[list[Any]] = []
    if limit and len(df):
        for record in df.head(limit).itertuples(index=False, name=None):
            rows.append([_preview_cell(item) for item in record])
    return {"headers": headers, "rows": rows}


def write_csv(df: pd.DataFrame, path: Any) -> int:
    """Write ``df`` as UTF-8 CSV without the index. Returns bytes written."""
    data = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if hasattr(path, "write"):
        try:
            path.write(data)
        except TypeError:
            path.write(data.decode("utf-8"))
    else:
        directory = os.path.dirname(os.fspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
    return len(data)
