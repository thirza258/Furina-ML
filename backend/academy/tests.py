"""Tests for accounts, saved progress, notebook launches and the hub's OAuth sign-in.

The OAuth tests drive the exact exchange JupyterHub performs: authorize with
PKCE as a signed-in student, swap the code for a token at the token endpoint,
then read the user-info endpoint with that token.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from io import StringIO
from urllib.parse import parse_qs, urlparse

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.management import call_command
from django.test import TestCase, override_settings
from oauth2_provider.models import get_access_token_model, get_application_model
from rest_framework.test import APIClient

from academy.models import LessonProgress, NotebookLaunch

PASSWORD = "a-long-enough-passphrase-41"
HUB = "http://notebooks.furina.test"
CLIENT_ID = "jupyterhub-client"
CLIENT_SECRET = "jupyterhub-secret-for-tests"


class AcademyTestCase(TestCase):
    """A client that enforces CSRF exactly as the browser-facing server does."""

    def setUp(self) -> None:
        # Throttle counters live in the cache, which outlives a single test.
        cache.clear()
        self.client = APIClient(enforce_csrf_checks=True)

    def csrf(self) -> str:
        self.client.get("/api/auth/session/")
        return self.client.cookies["csrftoken"].value

    def post(self, url: str, body: dict | None = None, *, token: bool = True):
        headers = {"HTTP_X_CSRFTOKEN": self.csrf()} if token else {}
        return self.client.post(url, body or {}, format="json", **headers)

    def make_user(self, username: str = "ada", **extra):
        return get_user_model().objects.create_user(username=username, password=PASSWORD, **extra)

    def sign_in(self, username: str = "ada") -> None:
        response = self.post("/api/auth/login/", {"username": username, "password": PASSWORD})
        self.assertEqual(response.status_code, 200, response.content)


class SessionTests(AcademyTestCase):
    def test_anonymous_session_sets_the_csrf_cookie(self) -> None:
        response = self.client.get("/api/auth/session/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("csrftoken", response.cookies)
        payload = response.json()
        self.assertFalse(payload["authenticated"])
        self.assertIsNone(payload["user"])
        self.assertTrue(payload["registration_open"])
        self.assertEqual(payload["notebooks"], {"hub_enabled": False, "hub_url": None})

    @override_settings(JUPYTERHUB_URL=HUB)
    def test_session_reports_the_hub(self) -> None:
        payload = self.client.get("/api/auth/session/").json()
        self.assertEqual(payload["notebooks"], {"hub_enabled": True, "hub_url": HUB})


class RegisterTests(AcademyTestCase):
    def test_register_requires_the_csrf_token(self) -> None:
        self.csrf()
        response = self.post(
            "/api/auth/register/", {"username": "grace", "password": PASSWORD}, token=False
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("CSRF", response.json()["detail"])
        self.assertFalse(get_user_model().objects.filter(username="grace").exists())

    def test_register_creates_a_signed_in_student(self) -> None:
        response = self.post(
            "/api/auth/register/",
            {"username": "grace", "email": "grace@example.com", "password": PASSWORD},
        )
        self.assertEqual(response.status_code, 201, response.content)
        payload = response.json()
        self.assertTrue(payload["authenticated"])
        self.assertEqual(payload["user"]["username"], "grace")
        self.assertEqual(payload["user"]["roles"], ["student"])
        self.assertTrue(self.client.get("/api/auth/session/").json()["authenticated"])

    def test_username_is_unique_ignoring_case(self) -> None:
        self.make_user("Grace")
        response = self.post("/api/auth/register/", {"username": "grace", "password": PASSWORD})
        self.assertEqual(response.status_code, 400)
        self.assertIn("username", response.json()["errors"])

    def test_weak_password_is_refused_with_reasons(self) -> None:
        response = self.post("/api/auth/register/", {"username": "grace", "password": "123"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("password", response.json()["errors"])

    @override_settings(ACCOUNT_REGISTRATION=False)
    def test_closed_registration(self) -> None:
        response = self.post("/api/auth/register/", {"username": "grace", "password": PASSWORD})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.client.get("/api/auth/session/").json()["registration_open"])


class LoginTests(AcademyTestCase):
    def test_wrong_password_and_unknown_user_look_the_same(self) -> None:
        self.make_user()
        wrong = self.post("/api/auth/login/", {"username": "ada", "password": "nope"})
        unknown = self.post("/api/auth/login/", {"username": "nobody", "password": "nope"})
        self.assertEqual(wrong.status_code, 400)
        self.assertEqual(wrong.json(), unknown.json())

    def test_login_then_logout(self) -> None:
        self.make_user()
        self.sign_in()
        self.assertTrue(self.client.get("/api/auth/session/").json()["authenticated"])
        response = self.post("/api/auth/logout/")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["authenticated"])

    def test_staff_users_carry_the_admin_role(self) -> None:
        self.make_user(is_staff=True)
        self.sign_in()
        user = self.client.get("/api/auth/session/").json()["user"]
        self.assertIn("admin", user["roles"])

    def test_sign_in_is_rate_limited(self) -> None:
        codes = [
            self.post("/api/auth/login/", {"username": "x", "password": "y"}).status_code
            for _ in range(11)
        ]
        self.assertEqual(codes[:10], [400] * 10)
        self.assertEqual(codes[10], 429)

    def test_signed_in_workbench_posts_need_the_csrf_token(self) -> None:
        self.make_user()
        self.sign_in()
        without = self.client.post("/api/train/", {}, format="json")
        self.assertEqual(without.status_code, 403)
        with_token = self.post("/api/train/", {})
        # Past the CSRF check: an empty body is now a validation error.
        self.assertEqual(with_token.status_code, 400)


class ProgressTests(AcademyTestCase):
    def test_progress_needs_an_account(self) -> None:
        self.assertEqual(self.client.get("/api/progress/").status_code, 403)

    def test_progress_round_trip_is_idempotent(self) -> None:
        self.make_user()
        self.sign_in()
        body = {"lessons": ["training/fit-a-model", "math-foundations/calculus-from-scratch"]}
        first = self.post("/api/progress/", body)
        self.assertEqual(first.status_code, 200, first.content)
        again = self.post("/api/progress/", {"lessons": ["training/fit-a-model"]})
        self.assertEqual(
            sorted(again.json()["completed"]),
            ["math-foundations/calculus-from-scratch", "training/fit-a-model"],
        )
        self.assertEqual(LessonProgress.objects.count(), 2)
        self.assertEqual(len(self.client.get("/api/progress/").json()["completed"]), 2)

    def test_malformed_keys_are_refused(self) -> None:
        self.make_user()
        self.sign_in()
        for key in ("training", "../etc/passwd", "Training/Fit", "a/b/c"):
            response = self.post("/api/progress/", {"lessons": [key]})
            self.assertEqual(response.status_code, 400, key)
        self.assertEqual(LessonProgress.objects.count(), 0)

    def test_students_only_see_their_own_progress(self) -> None:
        other = self.make_user("other")
        LessonProgress.objects.create(user=other, course_id="training", lesson_id="fit-a-model")
        self.make_user()
        self.sign_in()
        self.assertEqual(self.client.get("/api/progress/").json()["completed"], [])


class NotebookLaunchTests(AcademyTestCase):
    path = "math-foundations/01_python_math_foundations.ipynb"

    def test_launch_needs_an_account(self) -> None:
        response = self.post("/api/notebooks/launch/", {"path": self.path})
        self.assertEqual(response.status_code, 403)

    def test_launch_without_a_hub_is_unavailable(self) -> None:
        self.make_user()
        self.sign_in()
        response = self.post("/api/notebooks/launch/", {"path": self.path})
        self.assertEqual(response.status_code, 503)

    @override_settings(JUPYTERHUB_URL=HUB)
    def test_launch_returns_the_user_redirect_url(self) -> None:
        user = self.make_user()
        self.sign_in()
        response = self.post("/api/notebooks/launch/", {"path": self.path})
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(
            response.json()["url"], f"{HUB}/hub/user-redirect/notebooks/{self.path}"
        )
        self.assertEqual(
            response.json()["lab_url"], f"{HUB}/hub/user-redirect/lab/tree/{self.path}"
        )
        self.assertEqual(NotebookLaunch.objects.get().user, user)

    @override_settings(JUPYTERHUB_URL=HUB)
    def test_only_template_paths_are_accepted(self) -> None:
        self.make_user()
        self.sign_in()
        for path in ("../secret.ipynb", "/etc/passwd", "course/run.py", "a/b/c.ipynb", "x/..ipynb"):
            response = self.post("/api/notebooks/launch/", {"path": path})
            self.assertEqual(response.status_code, 400, path)
        self.assertFalse(NotebookLaunch.objects.exists())


@override_settings(
    JUPYTERHUB_URL=HUB,
    JUPYTERHUB_OAUTH_CLIENT_ID=CLIENT_ID,
    JUPYTERHUB_OAUTH_CLIENT_SECRET=CLIENT_SECRET,
)
class HubOAuthTests(AcademyTestCase):
    redirect_uri = f"{HUB}/hub/oauth_callback"

    def setUp(self) -> None:
        super().setUp()
        call_command("ensure_jupyterhub_client", stdout=StringIO())
        self.verifier = secrets.token_urlsafe(48)
        digest = hashlib.sha256(self.verifier.encode()).digest()
        self.challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def authorize(self):
        return self.client.get(
            "/api/oauth/authorize/",
            {
                "response_type": "code",
                "client_id": CLIENT_ID,
                "redirect_uri": self.redirect_uri,
                "scope": "profile",
                "state": "opaque-state",
                "code_challenge": self.challenge,
                "code_challenge_method": "S256",
            },
        )

    def exchange(self, code: str):
        # The hub calls this server to server, with no browser session.
        return APIClient().post(
            "/api/oauth/token/",
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code_verifier": self.verifier,
            },
        )

    def test_command_is_idempotent_and_hashes_the_secret(self) -> None:
        call_command("ensure_jupyterhub_client", stdout=StringIO())
        application = get_application_model().objects.get(client_id=CLIENT_ID)
        self.assertEqual(get_application_model().objects.count(), 1)
        self.assertTrue(application.skip_authorization)
        self.assertNotEqual(application.client_secret, CLIENT_SECRET)

    def test_signed_out_students_are_sent_to_the_login_page(self) -> None:
        response = self.authorize()
        self.assertEqual(response.status_code, 302)
        location = urlparse(response["Location"])
        self.assertEqual(location.path, "/login")
        self.assertTrue(parse_qs(location.query)["next"][0].startswith("/api/oauth/authorize/"))

    def test_full_hub_sign_in(self) -> None:
        user = self.make_user()
        self.sign_in()
        response = self.authorize()
        self.assertEqual(response.status_code, 302)
        location = urlparse(response["Location"])
        self.assertEqual(f"{location.scheme}://{location.netloc}{location.path}", self.redirect_uri)
        query = parse_qs(location.query)
        self.assertEqual(query["state"], ["opaque-state"])

        token = self.exchange(query["code"][0])
        self.assertEqual(token.status_code, 200, token.content)
        access = token.json()["access_token"]

        reader = APIClient()
        info = reader.get("/api/auth/hub-user/", HTTP_AUTHORIZATION=f"Bearer {access}")
        self.assertEqual(info.status_code, 200, info.content)
        self.assertEqual(
            info.json(),
            {"name": f"u{user.pk}", "username": "ada", "admin": False, "groups": []},
        )

        # Signing out of Furina revokes the hub's token.
        self.post("/api/auth/logout/")
        self.assertFalse(get_access_token_model().objects.exists())
        revoked = reader.get("/api/auth/hub-user/", HTTP_AUTHORIZATION=f"Bearer {access}")
        self.assertIn(revoked.status_code, (401, 403))

    def test_code_is_useless_without_the_pkce_verifier(self) -> None:
        self.make_user()
        self.sign_in()
        code = parse_qs(urlparse(self.authorize()["Location"]).query)["code"][0]
        self.verifier = secrets.token_urlsafe(48)
        self.assertEqual(self.exchange(code).status_code, 400)

    def test_user_info_needs_a_token(self) -> None:
        self.assertIn(self.client.get("/api/auth/hub-user/").status_code, (401, 403))
