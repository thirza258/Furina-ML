#!/bin/sh
# Copy course notebook templates the student does not have yet into their
# workspace, then start the notebook server. Files the student already has --
# edited or not -- are never overwritten; deleting one brings the original
# back on the next start.
set -eu

python3 - <<'PY'
import shutil
from pathlib import Path

source = Path("/opt/furina/notebooks")
workspace = Path.home() / "work"
copied = 0
for template in sorted(source.rglob("*")):
    if template.is_dir() or "__pycache__" in template.parts:
        continue
    target = workspace / template.relative_to(source)
    if target.exists():
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, target)
    copied += 1
print(f"[furina] {copied} course file(s) added to the workspace")
PY

exec "$@"
