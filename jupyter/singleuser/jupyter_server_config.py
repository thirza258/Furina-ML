"""Jupyter Server settings for a student's notebook server."""

import os

c = get_config()  # noqa: F821 - provided by Jupyter when it loads this file

# Only the Furina site (which embeds the notebook beside the lesson) and the
# server itself may frame these pages.
site_origin = os.environ.get("FURINA_SITE_ORIGIN", "").strip()
c.ServerApp.tornado_settings = {
    "headers": {"Content-Security-Policy": f"frame-ancestors 'self' {site_origin}".strip()}
}

# Free the memory of kernels nobody has used for half an hour. The notebook
# files are untouched; the hub's idle culler stops whole servers separately.
c.MappingKernelManager.cull_idle_timeout = 1800
c.MappingKernelManager.cull_interval = 300
