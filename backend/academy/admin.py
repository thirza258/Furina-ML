"""Django admin registrations for saved progress and notebook launches."""

from django.contrib import admin

from academy.models import LessonProgress, NotebookLaunch


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
