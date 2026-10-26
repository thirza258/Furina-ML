"""HTTP API for accounts, saved course progress and notebook launches.

Mounted under ``/api/`` next to the workbench endpoints, which stay open to
anonymous visitors. The SPA signs in with a Django session: it reads the CSRF
cookie set by ``GET /api/auth/session/`` and sends it back as ``X-CSRFToken``.
Sign-in, sign-up and sign-out check that token even for an anonymous caller,
so another site cannot sign a visitor in or out behind their back.
"""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.models import Group
from django.db import transaction
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from drf_spectacular.utils import extend_schema
from oauth2_provider.contrib.rest_framework import OAuth2Authentication, TokenHasScope
from oauth2_provider.models import get_access_token_model, get_refresh_token_model
from rest_framework import status
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from academy.hub import hub_enabled, hub_username, lab_url, launch_url
from academy.models import INSTRUCTOR_GROUP, STUDENT_GROUP, LessonProgress, NotebookLaunch
from academy.serializers import (
    LoginSerializer,
    NotebookLaunchSerializer,
    ProgressSerializer,
    RegisterSerializer,
)


def roles(user) -> list[str]:
    """The academy roles a user holds: student, instructor and/or admin."""
    names = set(user.groups.filter(name__in=[STUDENT_GROUP, INSTRUCTOR_GROUP]).values_list("name", flat=True))
    if user.is_staff:
        names.add("admin")
    return sorted(names)


def _user_payload(user) -> dict:
    return {
        "username": user.get_username(),
        "email": user.email,
        "roles": roles(user),
        "date_joined": user.date_joined.isoformat(),
    }


def _session_payload(request) -> dict:
    user = request.user
    return {
        "authenticated": bool(user.is_authenticated),
        "user": _user_payload(user) if user.is_authenticated else None,
        "registration_open": bool(settings.ACCOUNT_REGISTRATION),
        "notebooks": {
            "hub_enabled": hub_enabled(),
            "hub_url": settings.JUPYTERHUB_URL or None,
        },
    }


class _CsrfCheckedView(APIView):
    """An endpoint that checks the CSRF token even for anonymous callers.

    DRF only checks CSRF once a session is authenticated, which would leave
    sign-in itself open to login CSRF. The failure is raised as DRF's own
    PermissionDenied, so it comes back as the JSON error envelope.
    """

    def initial(self, request, *args, **kwargs):
        SessionAuthentication().enforce_csrf(request)
        super().initial(request, *args, **kwargs)


class SessionView(APIView):
    """Who is signed in, and what this deployment offers. Sets the CSRF cookie."""

    @extend_schema(responses=dict)
    @method_decorator(ensure_csrf_cookie)
    def get(self, request):
        return Response(_session_payload(request))


class RegisterView(_CsrfCheckedView):
    """Create a student account and sign it in."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth"

    @extend_schema(request=RegisterSerializer, responses=dict)
    def post(self, request):
        if not settings.ACCOUNT_REGISTRATION:
            return Response(
                {"detail": "New accounts are created by an administrator on this site."},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        with transaction.atomic():
            user = get_user_model().objects.create_user(
                username=data["username"],
                email=data.get("email", ""),
                password=data["password"],
            )
            user.groups.add(Group.objects.get_or_create(name=STUDENT_GROUP)[0])
        login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        return Response(_session_payload(request), status=status.HTTP_201_CREATED)


class LoginView(_CsrfCheckedView):
    """Sign in with a username and password."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth"

    @extend_schema(request=LoginSerializer, responses=dict)
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = authenticate(
            request,
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"],
        )
        if user is None:
            # One message for an unknown name and a wrong password alike, so
            # the form cannot be used to find out which usernames exist.
            return Response(
                {"detail": "That username and password do not match an account."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        login(request, user)
        return Response(_session_payload(request))


class LogoutView(_CsrfCheckedView):
    """Sign out, and end the notebook server's access with it.

    The hub re-checks its OAuth token against ``/api/auth/hub-user/``;
    deleting the tokens here makes that check fail, so the hub drops the
    student's session within its refresh interval.
    """

    @extend_schema(request=None, responses=dict)
    def post(self, request):
        user = request.user
        if user.is_authenticated:
            get_refresh_token_model().objects.filter(user=user).delete()
            get_access_token_model().objects.filter(user=user).delete()
        logout(request)
        return Response(_session_payload(request))


class HubUserView(APIView):
    """The OAuth user-info endpoint JupyterHub signs students in with."""

    authentication_classes = [OAuth2Authentication]
    permission_classes = [TokenHasScope]
    required_scopes = ["profile"]

    @extend_schema(responses=dict)
    def get(self, request):
        user = request.user
        return Response(
            {
                "name": hub_username(user),
                "username": user.get_username(),
                "admin": bool(user.is_staff),
                "groups": roles(user),
            }
        )


class ProgressView(APIView):
    """Passed knowledge checks for the signed-in student."""

    permission_classes = [IsAuthenticated]

    def _completed(self, request) -> list[str]:
        return [entry.key for entry in LessonProgress.objects.filter(user=request.user)]

    @extend_schema(responses=dict)
    def get(self, request):
        return Response({"completed": self._completed(request)})

    @extend_schema(request=ProgressSerializer, responses=dict)
    def post(self, request):
        """Mark lessons complete. Already-saved lessons are left as they are,
        so the browser can send everything it passed while signed out."""
        serializer = ProgressSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        LessonProgress.objects.bulk_create(
            [
                LessonProgress(user=request.user, course_id=course_id, lesson_id=lesson_id)
                for course_id, lesson_id in serializer.validated_data["lessons"]
            ],
            ignore_conflicts=True,
        )
        return Response({"completed": self._completed(request)})


class NotebookLaunchView(APIView):
    """Open a lesson notebook in the student's own JupyterHub workspace."""

    permission_classes = [IsAuthenticated]

    @extend_schema(request=NotebookLaunchSerializer, responses=dict)
    def post(self, request):
        if not hub_enabled():
            return Response(
                {
                    "detail": "The notebook server is not enabled on this site. Use the "
                    "in-browser preview instead."
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        serializer = NotebookLaunchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        path = serializer.validated_data["path"]
        NotebookLaunch.objects.create(user=request.user, path=path)
        return Response({"path": path, "url": launch_url(path), "lab_url": lab_url(path)})
