"""Capture the software environment a training run executed under.

Everything here is best-effort: any value we cannot determine is recorded as
``None`` rather than raising, because a missing version string must never be
the reason a run cannot be logged. The result is a JSON-safe dict that a
"reproduce this run" feature can diff against the environment it finds at
reproduction time, so it can honestly say whether a byte-exact reproduction is
possible.
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any

#: The app's own version, kept in one place. `settings` and the model card
#: both report it, so it is not a Django import here: a worker running outside
#: a configured Django project can still answer "which Furina trained this".
FURINA_VERSION = "1.0"

_MODULES = (
    "python",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "joblib",
)

# importlib uses the real module name; the key is what we report under.
_IMPORT_NAMES = {
    "scikit-learn": "sklearn",
}


def _import_version(name: str) -> str | None:
    module = _IMPORT_NAMES.get(name, name)
    try:
        candidate = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a missing dependency is recorded, not fatal
        return None
    version = getattr(candidate, "__version__", None)
    return str(version) if version else None


def _git_commit() -> str | None:
    """The checkout's commit when we can read one; None everywhere else.

    `git rev-parse` walks up the directory tree itself, so a working copy
    several levels deep still resolves to its own commit.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = (result.stdout or "").strip()
        return commit or None
    except (OSError, subprocess.SubprocessError):
        return None


def capture_environment() -> dict[str, Any]:
    """Return a JSON-safe snapshot of the running software stack."""
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "furina_version": FURINA_VERSION,
        "git_commit": _git_commit(),
        "packages": {name: _import_version(name) for name in _MODULES},
        "python_version": sys.version,  # full CPython string for the record
    }


def environment_diffs(captured: dict[str, Any] | None, current: dict[str, Any] | None) -> list[str]:
    """List the environment fields that differ between two snapshots.

    An empty list means nothing *we record* changed; the caller still must
    warn that binary reproducibility across library releases is not
    guaranteed.
    """
    if not captured or not current:
        return []
    captured = captured or {}
    current = current or {}
    diffs: list[str] = []
    for key in ("python", "furina_version", "git_commit"):
        if captured.get(key) != current.get(key):
            diffs.append(f"{key}: {captured.get(key)!r} -> {current.get(key)!r}")
    cap_packages = captured.get("packages") or {}
    cur_packages = current.get("packages") or {}
    for name in sorted(set(cap_packages) | set(cur_packages)):
        if cap_packages.get(name) != cur_packages.get(name):
            diffs.append(f"package {name}: {cap_packages.get(name)!r} -> {cur_packages.get(name)!r}")
    return diffs
