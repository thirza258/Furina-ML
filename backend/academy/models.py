"""Database models for the academy: saved progress and notebook launches.

Courses themselves are still defined in the frontend's course catalogue, so a
lesson is identified here by the same stable ``course_id`` and ``lesson_id``
strings that form its URL (``/app/courses/{course_id}/{lesson_id}``).
"""

from __future__ import annotations

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
