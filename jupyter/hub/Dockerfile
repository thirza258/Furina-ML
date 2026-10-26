# syntax=docker/dockerfile:1
#
# Furina ML -- JupyterHub. Signs students in through the Furina backend (OAuth),
# starts one isolated notebook container per student through a restricted
# Docker API proxy, and stops idle servers.

# The tag pins jupyterhub; a student server's jupyterhub-singleuser must match
# it (see jupyter/singleuser/requirements.txt).
FROM quay.io/jupyterhub/jupyterhub:6.1.1

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# The hub needs no root privileges: it reaches Docker over the socket proxy's
# TCP port, not the socket itself. Its database, cookie secret and proxy pid
# live in the hub_data volume mounted at /srv/jupyterhub/data.
RUN useradd --system --uid 990 --home-dir /srv/jupyterhub --shell /usr/sbin/nologin hub \
 && mkdir -p /srv/jupyterhub/data \
 && chown -R hub:hub /srv/jupyterhub

COPY jupyterhub_config.py /etc/jupyterhub/jupyterhub_config.py

USER hub
WORKDIR /srv/jupyterhub/data
EXPOSE 8000 8081

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8081/hub/health', timeout=4).read()" || exit 1

CMD ["jupyterhub", "-f", "/etc/jupyterhub/jupyterhub_config.py"]
