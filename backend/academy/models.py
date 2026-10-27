"""Database models for the academy: progress, notebook launches and submissions.

Courses themselves are still defined in the frontend's course catalogue, so a
lesson is identified here by the same stable ``course_id`` and ``lesson_id``
strings that form its URL (``/app/courses/{course_id}/{lesson_id}``).
"""

from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models

#: Roles are Django groups, created by a data migration. Staff status is the
#: administrator role.
STUDENT_GROUP = "student"
INSTRUCTOR_GROUP = "instructor"


class LessonProgress(models.Model):
    """One passed knowledge check, saved for a signed-in student."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="lesson_progress"
    )
    course_id = models.CharField(max_length=64)
    lesson_id = models.CharField(max_length=96)
    completed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "course_id", "lesson_id"], name="unique_lesson_progress"
            )
        ]
        ordering = ["completed_at"]
        verbose_name_plural = "lesson progress"

    def __str__(self) -> str:
        return f"{self.user} completed {self.key}"

    @property
    def key(self) -> str:
        """The ``course/lesson`` key the frontend stores progress under."""
        return f"{self.course_id}/{self.lesson_id}"


class NotebookLaunch(models.Model):
    """A student opened a lesson notebook on their JupyterHub server.

    Only the fact of the launch is recorded -- never the notebook's code or
    outputs -- which is what the launch counts per lesson need.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="notebook_launches"
    )
    path = models.CharField(max_length=255)
    launched_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-launched_at"]
        indexes = [models.Index(fields=["path", "launched_at"])]

    def __str__(self) -> str:
        return f"{self.user} opened {self.path}"


class Submission(models.Model):
    """A lesson notebook a student handed in for automated grading.

    Only what grading needs is stored: the code cells' ids and source, never
    outputs (which can be large and may hold data the student loaded). The
    grader worker claims queued submissions over the internal grader API, runs
    them in a sandbox and posts the result back; it never touches the database.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        GRADED = "graded", "Graded"
        ERROR = "error", "Could not be graded"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="submissions"
    )
    #: The lesson notebook template this is a copy of, e.g. ``neural-networks/02_build_a_neuron.ipynb``.
    path = models.CharField(max_length=255)
    notebook = models.JSONField()
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)
    #: Fraction of the notebook's checks that passed, 0 to 1.
    score = models.FloatField(null=True, blank=True)
    passed = models.PositiveIntegerField(null=True, blank=True)
    total = models.PositiveIntegerField(null=True, blank=True)
    #: Per-check outcomes and feedback, as the grader reported them.
    result = models.JSONField(default=dict, blank=True)
    #: Why grading failed, when status is ``error``. Shown to the student.
    error = models.TextField(blank=True)
    attempts = models.PositiveSmallIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "created_at"]),
            models.Index(fields=["user", "path", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.user} submitted {self.path} ({self.status})"
