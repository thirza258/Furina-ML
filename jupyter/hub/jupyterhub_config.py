"""JupyterHub configuration for Furina ML.

Every value comes from the environment, which docker-compose.yml fills from
`.env`. The design, and why each piece is here:

* **Sign-in** -- Furina's Django backend is the OAuth2 provider (authorization
  code + PKCE). A student who is signed in to Furina reaches their notebook
  without a second login. The hub re-checks its token every
  ``auth_refresh_age`` seconds; signing out of Furina revokes the token, so
  hub access ends within that interval.
* **Isolation** -- each student gets their own container: non-root, all Linux
  capabilities dropped, CPU, memory and process limits, on an internal Docker
  network with no route to the database, the Furina API or the internet.
  Every server is served from its own subdomain (``subdomain_host``), so one
  student's pages cannot read another student's cookies or call the hub as
  them; that is JupyterHub's documented isolation for untrusted users.
* **Docker access** -- the hub never touches the Docker socket. It talks to a
  socket proxy that only allows the container calls spawning needs. The hub
  is still a privileged component: whoever controls it can start containers.
* **Persistence** -- a student's files live in a named volume per student, so
  notebooks survive the container (and its kernel) being stopped or culled.
"""

from __future__ import annotations

import os
import sys

c = get_config()  # noqa: F821 - provided by JupyterHub when it loads this file


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is None:
        sys.exit(f"[furina] {name} must be set to start JupyterHub; see .env.example.")
    return default


site_url = env("SITE_URL").rstrip("/")
hub_url = env("JUPYTERHUB_URL").rstrip("/")
# How the hub reaches Django over the internal network (never the browser).
backend_url = env("FURINA_INTERNAL_URL", "http://backend:8000").rstrip("/")

# --------------------------------------------------------------- networking
# nginx in the frontend container forwards the hub's host name, and every
# student subdomain under it, to the proxy on port 8000.
c.JupyterHub.bind_url = "http://0.0.0.0:8000"
c.JupyterHub.hub_bind_url = "http://0.0.0.0:8081"
c.JupyterHub.hub_connect_url = env("JUPYTERHUB_CONNECT_URL", "http://jupyterhub:8081")
c.JupyterHub.subdomain_host = hub_url
c.JupyterHub.public_url = hub_url

# Hub pages, including the sign-in redirects, may be framed by the Furina
# site, which embeds the student's notebook next to the lesson.
c.JupyterHub.tornado_settings = {
    "headers": {"Content-Security-Policy": f"frame-ancestors 'self' {site_url}"},
}

# State lives in the hub_data volume.
c.JupyterHub.db_url = "sqlite:////srv/jupyterhub/data/jupyterhub.sqlite"
c.JupyterHub.cookie_secret_file = "/srv/jupyterhub/data/jupyterhub_cookie_secret"
c.ConfigurableHTTPProxy.pid_file = "/srv/jupyterhub/data/jupyterhub-proxy.pid"
c.JupyterHub.allow_named_servers = False
# Signing out of the hub also stops the student's server.
c.JupyterHub.shutdown_on_logout = True

# ------------------------------------------------------------------ sign-in
c.JupyterHub.authenticator_class = "generic-oauth"
c.GenericOAuthenticator.login_service = "Furina ML"
c.GenericOAuthenticator.client_id = env("JUPYTERHUB_OAUTH_CLIENT_ID")
c.GenericOAuthenticator.client_secret = env("JUPYTERHUB_OAUTH_CLIENT_SECRET")
c.GenericOAuthenticator.oauth_callback_url = f"{hub_url}/hub/oauth_callback"
# The browser is sent here; the next two are called server to server.
c.GenericOAuthenticator.authorize_url = f"{site_url}/api/oauth/authorize/"
c.GenericOAuthenticator.token_url = f"{backend_url}/api/oauth/token/"
c.GenericOAuthenticator.userdata_url = f"{backend_url}/api/auth/hub-user/"
c.GenericOAuthenticator.scope = ["profile"]
# "u<id>": stable, DNS-safe (it becomes a subdomain), no personal data in URLs.
c.GenericOAuthenticator.username_claim = "name"
c.GenericOAuthenticator.enable_pkce = True
# Every Furina account may have a notebook server; Furina decides who has one.
c.GenericOAuthenticator.allow_all = True
# Skip the hub's own "Sign in with Furina ML" button.
c.Authenticator.auto_login = True
# Furina's sign-out ends here: the hub deletes every token of the browser
# session (the student servers' too), stops the server, and sends the
# student back to the site.
c.GenericOAuthenticator.logout_redirect_url = f"{site_url}/app/courses"
# Keep the access token (encrypted with JUPYTERHUB_CRYPT_KEY) so the hub can
# re-check it, and re-check it before starting a server.
c.Authenticator.enable_auth_state = True
c.Authenticator.auth_refresh_age = int(env("JUPYTERHUB_AUTH_REFRESH_SECONDS", "300"))
c.Authenticator.refresh_pre_spawn = True
env("JUPYTERHUB_CRYPT_KEY")  # read by JupyterHub itself; fail early when missing

# ---------------------------------------------------------------- spawning
c.JupyterHub.spawner_class = "docker"
c.DockerSpawner.client_kwargs = {"base_url": env("DOCKER_HOST", "tcp://docker-proxy:2375")}
c.DockerSpawner.image = env("NOTEBOOK_IMAGE", "furina-notebook:latest")
# The image is built locally by docker compose; there is nothing to pull.
c.DockerSpawner.pull_policy = "never"
c.DockerSpawner.network_name = env("NOTEBOOK_NETWORK", "furina-students")
c.DockerSpawner.use_internal_ip = True
c.DockerSpawner.name_template = "furina-nb-{username}"
# The container is disposable; the student's files are not.
c.DockerSpawner.remove = True
c.DockerSpawner.notebook_dir = "/home/jovyan/work"
c.DockerSpawner.volumes = {"furina-nb-{username}": "/home/jovyan/work"}
c.DockerSpawner.mem_limit = env("NOTEBOOK_MEM_LIMIT", "1G")
c.DockerSpawner.cpu_limit = float(env("NOTEBOOK_CPU_LIMIT", "1"))
c.DockerSpawner.extra_host_config = {
    "pids_limit": int(env("NOTEBOOK_PIDS_LIMIT", "256")),
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges:true"],
}
c.DockerSpawner.environment = {"FURINA_SITE_ORIGIN": site_url}
c.Spawner.default_url = "/lab"
c.Spawner.start_timeout = 120
c.Spawner.http_timeout = 60

# ------------------------------------------------------------- idle culling
# Stops servers with no activity, which also frees their memory. The files
# stay in the student's volume; only the kernel's in-memory state is lost.
idle_seconds = int(env("NOTEBOOK_IDLE_SECONDS", "1800"))
c.JupyterHub.services = [
    {
        "name": "idle-culler",
        "command": [
            sys.executable,
            "-m",
            "jupyterhub_idle_culler",
            f"--timeout={idle_seconds}",
            "--cull-every=60",
        ],
    }
]
c.JupyterHub.load_roles = [
    {
        "name": "idle-culler",
        "scopes": ["list:users", "read:users:activity", "read:servers", "delete:servers"],
        "services": ["idle-culler"],
    }
]
