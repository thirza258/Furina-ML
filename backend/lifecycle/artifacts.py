"""The ArtifactStore abstraction from the PRD.

Artifacts (datasets, models, evaluation reports, prediction outputs, model
cards) all go through one interface so the deployment backend is
interchangeable. Today only the local filesystem storage that Django already
uses is implemented; ``S3ArtifactStore`` and ``MinIOArtifactStore`` plug into
the same interface when an operator wants object storage, and the application
code never learns which one is behind it.

The local implementation delegates to Django's storage so it honours the
``MEDIA_ROOT`` / volume configuration that the container deployment already
tunes. Nothing in this module is importable without Django.
"""

from __future__ import annotations

import io
from typing import Protocol

from django.core.files.base import ContentFile
from django.db import models


class ArtifactStore(Protocol):
    """The contract every storage backend implements."""

    def store_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Persist ``data`` under ``key`` and return the address to read it back."""
        ...

    def open(self, address: str) -> io.BytesIO:
        """Return the bytes stored at ``address``."""
        ...

    def exists(self, address: str) -> bool: ...

    def delete(self, address: str) -> None: ...


class LocalFilesystemArtifactStore:
    """Stores under ``media/artifacts/<kind>/`` via Django's default storage.

    The address it returns is the *storage name* (e.g.
    ``artifacts/models/03f2.joblib``). The file field of a model record is the
    source of truth for a run; this store is the path a deployment or batch
    endpoint uses to find a file whose row may no longer have a file field,
    and the place an S3 backend takes over without changing the callers.
    """

    ROOT = "artifacts"

    def __init__(self, storage: models.storage.files.FileSystemStorage | None = None) -> None:
        from django.core.files.storage import default_storage

        self._storage = storage or default_storage

    def _key(self, kind: str, data: bytes) -> str:
        import hashlib

        digest = hashlib.sha256(data).hexdigest()[:16]
        # The kind comes from the caller, not the user, so no sanitising is
        # strictly needed, but the name itself may not climb out of the root.
        safe_kind = kind.replace("..", "").replace("/", "").strip() or "misc"
        return f"{self.ROOT}/{safe_kind}/{digest}"

    def store_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        kind, _, name = key.partition("/")
        storage_name = f"{self.ROOT}/{kind or 'misc'}/{name}"
        self._storage.save(storage_name, ContentFile(data))
        return self._storage.name(storage_name)

    def open(self, address: str) -> io.BytesIO:
        with self._storage.open(address, "rb") as handle:
            return io.BytesIO(handle.read())

    def exists(self, address: str) -> bool:
        return self._storage.exists(address)

    def delete(self, address: str) -> None:
        if self._storage.exists(address):
            self._storage.delete(address)


class S3ArtifactStore:  # pragma: no cover - no S3 in the reference deployment
    """Placeholder for an S3/MinIO backend.

    Implements the same protocol so a settings-driven factory can return it;
    the constructor takes an already-configured ``boto3``-style client and a
    bucket, and ``address`` is an S3 object key. Raising at use time when the
    operator has not installed the SDK keeps the import above safe.
    """

    def __init__(self, bucket: str, client=None) -> None:
        self._bucket = bucket
        self._client = client

    def _require_client(self) -> None:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")

    def store_bytes(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        self._require_client()
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def open(self, address: str) -> io.BytesIO:
        self._require_client()
        obj = self._client.get_object(Bucket=self._bucket, Key=address)
        return io.BytesIO(obj["Body"].read())

    def exists(self, address: str) -> bool:
        self._require_client()
        try:
            self._client.head_object(Bucket=self._bucket, Key=address)
            return True
        except Exception:  # noqa: BLE001 - any S3 error means "not there"
            return False

    def delete(self, address: str) -> None:
        self._require_client()
        self._client.delete_object(Bucket=self._bucket, Key=address)


def get_default_store() -> LocalFilesystemArtifactStore:
    """The store every endpoint that does not name one uses."""
    return LocalFilesystemArtifactStore()
