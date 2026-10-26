"""How a Django account maps onto JupyterHub.

JupyterHub signs students in through this app's OAuth provider and gives each
one an isolated notebook server. Nothing here talks to the hub: launching a
notebook is a browser redirect through the hub's ``user-redirect`` URL, which
signs the student in (if needed), starts their server (if stopped) and opens
the file in their own workspace.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from django.conf import settings

#: A notebook template path as the lesson catalogue writes it:
#: ``<course-folder>/<file>.ipynb``. Anything else -- absolute paths, ``..``,
#: other file types -- is refused before it reaches a URL.
NOTEBOOK_PATH = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*/[A-Za-z0-9][A-Za-z0-9_.-]*\.ipynb$")


def hub_enabled() -> bool:
    """True when a JupyterHub address is configured."""
    return bool(settings.JUPYTERHUB_URL)


def hub_username(user) -> str:
    """The name a Django user has on the hub.

    Derived from the primary key rather than the username: it never changes,
    it is always a valid DNS label (each student's server has its own
    subdomain), and it keeps names and email addresses out of hub URLs.
    """
    return f"u{user.pk}"


def is_notebook_path(path: str) -> bool:
    """True for a well-formed ``<course>/<file>.ipynb`` template path."""
    return bool(NOTEBOOK_PATH.match(path)) and ".." not in path


def launch_url(path: str) -> str:
    """The hub URL that opens ``path`` in the signed-in student's workspace.

    It opens Jupyter Notebook's single-document view, which reads like a
    Colab notebook rather than JupyterLab's full IDE.
    """
    return f"{settings.JUPYTERHUB_URL}/hub/user-redirect/notebooks/{quote(path)}"


def lab_url(path: str) -> str:
    """The same notebook in the full JupyterLab interface."""
    return f"{settings.JUPYTERHUB_URL}/hub/user-redirect/lab/tree/{quote(path)}"
