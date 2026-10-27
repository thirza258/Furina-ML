"""Django admin registrations for saved progress, notebook launches and submissions."""

from django.contrib import admin

from academy.models import LessonProgress, NotebookLaunch, Submission


@admin.register(LessonProgress)
class LessonProgressAdmin(admin.ModelAdmin):
    """Which knowledge checks each student has passed."""

    list_display = ("user", "course_id", "lesson_id", "completed_at")
    list_filter = ("course_id", "completed_at")
    search_fields = ("user__username", "course_id", "lesson_id")
    ordering = ("-completed_at",)


@admin.register(NotebookLaunch)
class NotebookLaunchAdmin(admin.ModelAdmin):
    """Notebook openings on JupyterHub, for launch counts per lesson."""

    list_display = ("user", "path", "launched_at")
    list_filter = ("launched_at",)
    search_fields = ("user__username", "path")
    ordering = ("-launched_at",)


@admin.register(Submission)
class SubmissionAdmin(admin.ModelAdmin):
    """Graded notebooks, for instructors reviewing a student's work."""

    list_display = ("user", "path", "status", "score_percent", "created_at")
    list_filter = ("status", "path", "created_at")
    search_fields = ("user__username", "path")
    ordering = ("-created_at",)
    readonly_fields = (
        "id", "user", "path", "status", "score", "passed", "total", "result", "error",
        "attempts", "created_at", "started_at", "finished_at", "notebook",
    )
    actions = ["grade_again"]

    @admin.display(description="score", ordering="score")
    def score_percent(self, submission: Submission) -> str:
        if submission.score is None:
            return "-"
        return f"{submission.score:.0%} ({submission.passed}/{submission.total})"

    @admin.action(description="Grade the selected submissions again")
    def grade_again(self, request, queryset):
        count = queryset.exclude(status=Submission.Status.RUNNING).update(
            status=Submission.Status.QUEUED, attempts=0, error="", started_at=None,
        )
        self.message_user(request, f"{count} submission(s) queued for grading.")

    def has_add_permission(self, request) -> bool:
        return False
