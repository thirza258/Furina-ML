"""HTTP API for accounts, saved course progress, notebook launches and grading.

Mounted under ``/api/`` next to the workbench endpoints, which stay open to
anonymous visitors. The SPA signs in with a Django session: it reads the CSRF
cookie set by ``GET /api/auth/session/`` and sends it back as ``X-CSRFToken``.
Sign-in, sign-up and sign-out check that token even for an anonymous caller,
so another site cannot sign a visitor in or out behind their back.
"""

from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.models import Group
from django.db import transaction
from django.utils import timezone
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

from academy.grading import HasGraderToken, InvalidNotebook, code_cells, grading_enabled
from academy.hub import hub_enabled, hub_username, lab_url, launch_url
from academy.models import (
    INSTRUCTOR_GROUP,
    STUDENT_GROUP,
    LessonProgress,
    NotebookLaunch,
    Submission,
)
from academy.serializers import (
    GradeResultSerializer,
    LoginSerializer,
    NotebookLaunchSerializer,
    ProgressSerializer,
    RegisterSerializer,
    SubmissionCreateSerializer,
    SubmissionSerializer,
)

#: How many times a submission is tried before it is reported as an error.
MAX_GRADING_ATTEMPTS = 3


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
            "grading_enabled": grading_enabled(),
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


class SubmissionListView(APIView):
    """The signed-in student's submissions; POST hands a notebook in for grading."""

    permission_classes = [IsAuthenticated]
    throttle_scope = "submissions"

    def get_throttles(self):
        # Only handing in is rate limited; listing is what the page polls.
        return [ScopedRateThrottle()] if self.request.method == "POST" else []

    @extend_schema(responses=SubmissionSerializer(many=True))
    def get(self, request):
        submissions = Submission.objects.filter(user=request.user)
        path = request.query_params.get("path")
        if path:
            submissions = submissions.filter(path=path)
        return Response({"results": SubmissionSerializer(submissions[:20], many=True).data})

    @extend_schema(request=SubmissionCreateSerializer, responses=SubmissionSerializer)
    def post(self, request):
        if not grading_enabled():
            return Response(
                {"detail": "Automatic grading is not enabled on this site."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        if int(request.META.get("CONTENT_LENGTH") or 0) > settings.MAX_SUBMISSION_BYTES:
            return Response(
                {"detail": "That notebook is too large to grade. Clear its outputs and try again."},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        serializer = SubmissionCreateSerializer(data={"path": request.data.get("path")})
        serializer.is_valid(raise_exception=True)
        path = serializer.validated_data["path"]

        upload = request.FILES.get("notebook")
        try:
            notebook = code_cells(upload.read() if upload is not None else request.data.get("notebook"))
        except InvalidNotebook as error:
            return Response({"notebook": [str(error)]}, status=status.HTTP_400_BAD_REQUEST)

        pending = Submission.objects.filter(
            user=request.user,
            path=path,
            status__in=[Submission.Status.QUEUED, Submission.Status.RUNNING],
        )
        if pending.exists():
            return Response(
                {"detail": "Your last submission of this notebook is still being graded."},
                status=status.HTTP_409_CONFLICT,
            )
        submission = Submission.objects.create(user=request.user, path=path, notebook=notebook)
        return Response(SubmissionSerializer(submission).data, status=status.HTTP_201_CREATED)


class SubmissionDetailView(APIView):
    """One of the signed-in student's submissions, with its per-check results."""

    permission_classes = [IsAuthenticated]

    @extend_schema(responses=SubmissionSerializer)
    def get(self, request, submission_id):
        submission = Submission.objects.filter(id=submission_id, user=request.user).first()
        if submission is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(SubmissionSerializer(submission).data)


class GraderClaimView(APIView):
    """Internal: the grader worker takes the oldest queued submission.

    Answers 204 when there is nothing to grade. A submission left running
    longer than ``GRADING_STALE_SECONDS`` (its worker died) goes back in the
    queue, up to three attempts.
    """

    authentication_classes = []
    permission_classes = [HasGraderToken]

    @extend_schema(exclude=True)
    def post(self, request):
        now = timezone.now()
        stale = Submission.objects.filter(
            status=Submission.Status.RUNNING,
            started_at__lt=now - timedelta(seconds=settings.GRADING_STALE_SECONDS),
        )
        stale.filter(attempts__gte=MAX_GRADING_ATTEMPTS).update(
            status=Submission.Status.ERROR,
            error="Grading did not finish. Try submitting again.",
            finished_at=now,
        )
        stale.update(status=Submission.Status.QUEUED)

        with transaction.atomic():
            submission = (
                Submission.objects.select_for_update(skip_locked=True)
                .filter(status=Submission.Status.QUEUED)
                .order_by("created_at")
                .first()
            )
            if submission is None:
                return Response(status=status.HTTP_204_NO_CONTENT)
            submission.status = Submission.Status.RUNNING
            submission.started_at = now
            submission.attempts += 1
            submission.save(update_fields=["status", "started_at", "attempts"])
        return Response(
            {"id": str(submission.id), "path": submission.path, "notebook": submission.notebook}
        )


class GraderResultView(APIView):
    """Internal: the grader worker reports the outcome of a submission it claimed."""

    authentication_classes = []
    permission_classes = [HasGraderToken]

    @extend_schema(exclude=True)
    def post(self, request, submission_id):
        serializer = GradeResultSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        updated = Submission.objects.filter(
            id=submission_id, status=Submission.Status.RUNNING
        ).update(
            status=data["status"],
            score=data.get("score"),
            passed=data.get("passed"),
            total=data.get("total"),
            result=data.get("result", {}),
            error=data.get("error", ""),
            finished_at=timezone.now(),
        )
        if not updated:
            return Response(
                {"detail": "That submission is not being graded."}, status=status.HTTP_409_CONFLICT
            )
        return Response(status=status.HTTP_204_NO_CONTENT)
