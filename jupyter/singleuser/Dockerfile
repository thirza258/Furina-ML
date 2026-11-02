# syntax=docker/dockerfile:1
#
# Furina ML -- one student's notebook server, started by JupyterHub.
#
# Built with the course notebooks as an extra build context:
#   docker compose build notebook-image
# (docker-compose.yml passes `notebooks=./notebooks`).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The student account. JupyterHub's volume for this student is mounted on
# ~/work; creating it here with the right owner is what makes a brand-new
# volume writable by the student.
RUN useradd --create-home --uid 1000 --shell /bin/bash jovyan \
 && mkdir -p /home/jovyan/work \
 && chown jovyan:jovyan /home/jovyan/work

# A C and a C++ compiler for the course that builds its model in C, C++ and
# CUDA (notebooks/cuda-base-models). Student servers have no GPU, so its CUDA
# kernels run on the notebooks' CPU emulator here. Installed in the image
# because student servers have no internet access; the grader runs
# submissions in this image too.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ libc6-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# Notebooks declare the kernel "python": JupyterLite's name for its Pyodide
# kernel. On this server jupyter_client treats "python" as an alias of the
# stock "python3" kernel, so the same templates open here without a prompt.
RUN pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# Student servers have no internet access, so JupyterLab's news and update
# checks could only fail; turn them off instead of asking every student.
COPY overrides.json /usr/local/share/jupyter/lab/settings/overrides.json

COPY jupyter_server_config.py /etc/jupyter/jupyter_server_config.py
COPY --chmod=755 start.sh /usr/local/bin/furina-start

# The canonical course templates. Each student's server copies the ones it is
# missing into their own workspace at start-up; these originals never change
# underneath a student.
COPY --from=notebooks . /opt/furina/notebooks
RUN rm -rf /opt/furina/notebooks/tools /opt/furina/notebooks/README.md

USER jovyan
WORKDIR /home/jovyan/work
ENTRYPOINT ["furina-start"]
CMD ["jupyterhub-singleuser"]
