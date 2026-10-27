"""The grader worker: takes submissions from Django and grades each in a sandbox.

    python -m furina_grader.worker

The loop: claim the oldest queued submission over Django's internal grader
API, rebuild the lesson notebook from its template with the student's exercise
cells swapped in (`build.py`), run it in a fresh container of the student
image, score the recorded checks and post the result back.

The sandbox container has no network at all, a memory, CPU and process limit,
no Linux capabilities, and runs as the image's unprivileged user. It is
created per submission and removed afterwards. The worker reaches Docker only
through the socket proxy, and Django only through the internal API.

Configuration (environment):

    GRADER_TOKEN         shared secret for Django's grader API (required)
    GRADER_API_URL       Django's API, default http://backend:8000/api
    GRADER_IMAGE         the student image, default furina-notebook:latest
    GRADER_TEMPLATES     lesson templates, default /opt/furina/notebooks
    GRADER_CELL_TIMEOUT  seconds one cell may run, default 60
    GRADER_TIMEOUT       seconds one whole notebook may run, default 300
    GRADER_MEM_LIMIT     sandbox memory, default 1g
    GRADER_CPU_LIMIT     sandbox CPUs, default 1
    GRADER_POLL_SECONDS  wait between empty polls, default 3
    DOCKER_HOST          the Docker API, e.g. tcp://docker-proxy:2375
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import signal
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import docker
from docker.errors import DockerException
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import ReadTimeout

from furina_grader.build import build_grading_notebook, score

log = logging.getLogger("furina.grader")

#: Same rule as Django's `academy.hub.NOTEBOOK_PATH`: ``<course>/<file>.ipynb``.
NOTEBOOK_PATH = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*/[A-Za-z0-9][A-Za-z0-9_.-]*\.ipynb$")
#: Where the notebook is placed and run inside the sandbox.
SANDBOX_DIR = "/tmp/grade"
#: The student image's unprivileged user (see jupyter/singleuser/Dockerfile).
SANDBOX_UID = 1000
#: The largest executed notebook read back from a sandbox.
MAX_EXECUTED_BYTES = 32 * 1024 * 1024
EXECUTE_SCRIPT = Path(__file__).with_name("execute.py")


class Settings:
    def __init__(self, environ=os.environ) -> None:
        self.token = environ.get("GRADER_TOKEN", "").strip()
        self.api_url = environ.get("GRADER_API_URL", "http://backend:8000/api").rstrip("/")
        self.image = environ.get("GRADER_IMAGE", "furina-notebook:latest")
        self.templates = Path(environ.get("GRADER_TEMPLATES", "/opt/furina/notebooks"))
        self.cell_timeout = int(environ.get("GRADER_CELL_TIMEOUT", "60"))
        self.timeout = int(environ.get("GRADER_TIMEOUT", "300"))
        self.mem_limit = environ.get("GRADER_MEM_LIMIT", "1g")
        self.cpus = float(environ.get("GRADER_CPU_LIMIT", "1"))
        self.poll_seconds = float(environ.get("GRADER_POLL_SECONDS", "3"))


class GradingError(Exception):
    """A submission cannot be graded; the message is shown to the student."""


# --------------------------------------------------------------- Django's API


def _call(settings: Settings, method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{settings.api_url}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {settings.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status == 204:
            return None
        return json.loads(response.read())


def claim(settings: Settings) -> dict | None:
    return _call(settings, "POST", "/grader/claim/", {})


def report(settings: Settings, submission_id: str, outcome: dict) -> None:
    _call(settings, "POST", f"/grader/submissions/{submission_id}/result/", outcome)


# ------------------------------------------------------------------- grading


def load_template(settings: Settings, path: str) -> tuple[dict, Path]:
    if not NOTEBOOK_PATH.match(path) or ".." in path:
        raise GradingError("That is not a lesson notebook path.")
    template = (settings.templates / path).resolve()
    if settings.templates.resolve() not in template.parents or not template.is_file():
        raise GradingError(f"There is no lesson notebook at {path}.")
    return json.loads(template.read_text(encoding="utf-8")), template.parent


def _bundle(notebook: dict, folder: Path) -> bytes:
    """A tar of the sandbox's working folder: the notebook, the runner and the
    lesson folder's support files (helper modules), owned by the sandbox user."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        def add(name: str, content: bytes) -> None:
            info = tarfile.TarInfo(f"grade/{name}")
            info.size, info.mode = len(content), 0o644
            info.uid = info.gid = SANDBOX_UID
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(content))

        directory = tarfile.TarInfo("grade")
        directory.type, directory.mode = tarfile.DIRTYPE, 0o755
        directory.uid = directory.gid = SANDBOX_UID
        directory.mtime = int(time.time())
        tar.addfile(directory)
        for support in sorted(folder.iterdir()):
            if support.is_file() and support.suffix != ".ipynb":
                add(support.name, support.read_bytes())
        add("execute.py", EXECUTE_SCRIPT.read_bytes())
        add("grading.ipynb", json.dumps(notebook).encode())
    return buffer.getvalue()


def _read_back(container) -> dict | None:
    """The executed notebook from the sandbox, or None when it never got written."""
    try:
        stream, _ = container.get_archive(f"{SANDBOX_DIR}/executed.ipynb")
    except DockerException:
        return None
    raw = io.BytesIO()
    for chunk in stream:
        raw.write(chunk)
        if raw.tell() > MAX_EXECUTED_BYTES + 1024 * 1024:
            raise GradingError("The notebook printed too much output to grade.")
    raw.seek(0)
    with tarfile.open(fileobj=raw) as tar:
        member = tar.next()
        if member is None or not member.isfile() or member.size > MAX_EXECUTED_BYTES:
            return None
        handle = tar.extractfile(member)
        try:
            return json.loads(handle.read()) if handle else None
        except ValueError:
            return None


def run_in_sandbox(settings: Settings, client, notebook: dict, folder: Path) -> dict | None:
    """Run a grading notebook in a throwaway container; the executed notebook comes back."""
    memory = settings.mem_limit
    container = client.containers.create(
        settings.image,
        entrypoint=["python", f"{SANDBOX_DIR}/execute.py"],
        command=[
            f"{SANDBOX_DIR}/grading.ipynb",
            f"{SANDBOX_DIR}/executed.ipynb",
            str(settings.cell_timeout),
        ],
        working_dir=SANDBOX_DIR,
        user=str(SANDBOX_UID),
        network_mode="none",
        mem_limit=memory,
        memswap_limit=memory,
        nano_cpus=int(settings.cpus * 1e9),
        pids_limit=256,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        labels={"furina.grader": "sandbox"},
        environment={"MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    try:
        # Files go onto the container's own writable layer, before it starts:
        # the archive API cannot write into tmpfs mounts.
        container.put_archive("/tmp", _bundle(notebook, folder))
        container.start()
        try:
            container.wait(timeout=settings.timeout)
        except (ReadTimeout, RequestsConnectionError):
            # The forced remove below stops it; this is only to stop it sooner.
            try:
                container.kill()
            except DockerException:
                pass
            raise GradingError(
                f"Grading stopped after {settings.timeout} seconds. Check that no cell "
                "runs for a long time (an endless loop, a very long training run)."
            ) from None
        return _read_back(container)
    finally:
        try:
            container.remove(force=True)
        except DockerException:
            log.warning("could not remove sandbox %s", container.id[:12])


def grade(settings: Settings, client, job: dict) -> dict:
    """Grade one claimed submission; returns what to post back to Django."""
    try:
        template, folder = load_template(settings, str(job.get("path", "")))
        plan = build_grading_notebook(template, job.get("notebook") or {})
        if not plan.items:
            raise GradingError("This notebook has no graded exercises.")
        executed = run_in_sandbox(settings, client, plan.notebook, folder)
        if executed is None:
            raise GradingError(
                "The notebook stopped before it could be graded. It may have run out of "
                "memory; check for very large arrays or endless loops."
            )
    except GradingError as error:
        return {"status": "error", "error": str(error)}
    result = score(plan, executed)
    return {
        "status": "graded",
        "score": result["score"],
        "passed": result["passed"],
        "total": result["total"],
        "result": {
            key: result[key] for key in ("items", "answered", "unanswered", "complete")
        },
    }


# ---------------------------------------------------------------------- loop


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    if not settings.token:
        raise SystemExit("GRADER_TOKEN is not set: the grader has nothing to authenticate with.")
    client = docker.from_env(timeout=settings.timeout + 30)
    # Sandboxes left behind by a worker that was killed mid-run.
    for leftover in client.containers.list(all=True, filters={"label": "furina.grader=sandbox"}):
        leftover.remove(force=True)

    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    log.info("grader started: image %s, API %s", settings.image, settings.api_url)
    while not stopping:
        try:
            job = claim(settings)
        except (urllib.error.URLError, OSError, ValueError) as error:
            log.warning("cannot reach the grader API: %s", error)
            time.sleep(max(settings.poll_seconds, 10))
            continue
        if job is None:
            time.sleep(settings.poll_seconds)
            continue

        started = time.monotonic()
        try:
            outcome = grade(settings, client, job)
        except Exception:  # Docker trouble: not the student's fault
            log.exception("grading %s failed", job.get("id"))
            outcome = {"status": "error", "error": "The grader had a problem. Try again later."}
        log.info(
            "graded %s (%s): %s %s/%s in %.1fs",
            job.get("id"), job.get("path"), outcome["status"],
            outcome.get("passed"), outcome.get("total"), time.monotonic() - started,
        )
        try:
            report(settings, str(job["id"]), outcome)
        except (urllib.error.URLError, OSError, ValueError) as error:
            # The submission stays "running"; Django re-queues it once it is stale.
            log.warning("could not report %s: %s", job.get("id"), error)


if __name__ == "__main__":
    main()
