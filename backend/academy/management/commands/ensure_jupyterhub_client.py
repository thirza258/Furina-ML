"""Register (or refresh) the OAuth client JupyterHub signs students in with.

Runs on every container start. It does nothing unless JUPYTERHUB_URL,
JUPYTERHUB_OAUTH_CLIENT_ID and JUPYTERHUB_OAUTH_CLIENT_SECRET are all set, and
it is idempotent: the client keeps its id and gets the current secret and
redirect URI, so rotating the secret in `.env` takes effect on the next boot.
"""

from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand
from oauth2_provider.models import get_application_model

CLIENT_NAME = "JupyterHub"


class Command(BaseCommand):
    help = "Create or update the OAuth client used by JupyterHub."

    def handle(self, *args, **options) -> None:
        hub = settings.JUPYTERHUB_URL
        client_id = settings.JUPYTERHUB_OAUTH_CLIENT_ID
        secret = settings.JUPYTERHUB_OAUTH_CLIENT_SECRET
        if not (hub and client_id and secret):
            self.stdout.write("JupyterHub is not configured; no OAuth client registered.")
            return

        Application = get_application_model()
        application, created = Application.objects.get_or_create(
            client_id=client_id,
            defaults={"name": CLIENT_NAME},
        )
        application.name = CLIENT_NAME
        application.client_type = Application.CLIENT_CONFIDENTIAL
        application.authorization_grant_type = Application.GRANT_AUTHORIZATION_CODE
        application.redirect_uris = f"{hub}/hub/oauth_callback"
        # A first-party client: signing in to the hub must not stop at a
        # "do you allow this app?" page the student never asked to see.
        application.skip_authorization = True
        # Stored hashed, so the database never holds the usable secret.
        application.client_secret = secret
        application.full_clean(exclude=["user"])
        application.save()
        verb = "Registered" if created else "Updated"
        self.stdout.write(f"{verb} the OAuth client for {hub}.")
