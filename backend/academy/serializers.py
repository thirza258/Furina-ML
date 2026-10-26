"""Request validation for the academy endpoints."""

from __future__ import annotations

import re

from django.contrib.auth import get_user_model, password_validation
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import serializers

from academy.hub import is_notebook_path

#: Course and lesson ids are kebab-case slugs; they are also URL segments.
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_PROGRESS_KEYS = 500


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(
        max_length=150, validators=[UnicodeUsernameValidator()]
    )
    email = serializers.EmailField(required=False, allow_blank=True)
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate_username(self, value: str) -> str:
        if get_user_model().objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("That username is already taken.")
        return value

    def validate(self, attrs: dict) -> dict:
        candidate = get_user_model()(username=attrs["username"], email=attrs.get("email", ""))
        try:
            password_validation.validate_password(attrs["password"], user=candidate)
        except DjangoValidationError as exc:
            raise serializers.ValidationError({"password": list(exc.messages)}) from None
        return attrs


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, trim_whitespace=False)


def _split_key(key: str) -> tuple[str, str]:
    course_id, _, lesson_id = str(key).partition("/")
    if (
        not _SLUG.match(course_id)
        or not _SLUG.match(lesson_id)
        or len(course_id) > 64
        or len(lesson_id) > 96
    ):
        raise serializers.ValidationError(
            f"'{key}' is not a lesson key; use '<course-id>/<lesson-id>'."
        )
    return course_id, lesson_id


class ProgressSerializer(serializers.Serializer):
    """Lessons to mark complete, as ``course/lesson`` keys."""

    lessons = serializers.ListField(
        child=serializers.CharField(max_length=161), max_length=MAX_PROGRESS_KEYS
    )

    def validate_lessons(self, value: list[str]) -> list[tuple[str, str]]:
        return sorted({_split_key(key) for key in value})


class NotebookLaunchSerializer(serializers.Serializer):
    path = serializers.CharField(max_length=255)

    def validate_path(self, value: str) -> str:
        if not is_notebook_path(value):
            raise serializers.ValidationError(
                "Use a lesson notebook path such as 'math-foundations/01_intro.ipynb'."
            )
        return value
