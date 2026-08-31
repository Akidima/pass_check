"""Opt-in capture of inference examples for later, human-reviewed training.

This module deliberately does not train or mutate a live model. Capturing a
student image is a separate, explicitly enabled operation so inference stays
predictable and a bad model update cannot affect current submissions.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.environ.get("TRAINING_CAPTURE_ENABLED", "0").strip().lower() in (
        "1", "true", "yes",
    )


def capture_inference_example(
    image_bytes: bytes,
    result: dict[str, Any],
    *,
    identity_id: str | None = None,
    attire_policy: str | None = None,
    file_extension: str = ".jpg",
) -> str | None:
    """Persist an original image and inference metadata when explicitly enabled.

    The caller must provide consent before invoking this function. Files are
    written atomically into a server-configured directory. Missing identity
    metadata is retained as ``null`` and must not be used for model approval.
    Returns the capture ID, or ``None`` when capture is disabled.
    """
    if not _enabled():
        return None
    if not image_bytes:
        raise ValueError("Cannot capture an empty image.")

    capture_id = secrets.token_hex(16)
    try:
        capture_dir = Path(
            os.environ.get("TRAINING_CAPTURE_DIR", "training-data/inbox")
        )
        capture_dir.mkdir(parents=True, exist_ok=True)
        suffix = file_extension.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        image_path = capture_dir / f"{capture_id}{suffix}"
        metadata_path = capture_dir / f"{capture_id}.json"

        metadata = {
            "schema_version": 1,
            "capture_id": capture_id,
            "identity_id": identity_id,
            "attire_policy": attire_policy,
            "source": "production_inference",
            "result": result,
            "label": None,
            "label_source": None,
        }
        _atomic_write(image_path, image_bytes)
        _atomic_write(
            metadata_path,
            json.dumps(metadata, ensure_ascii=True, separators=(",", ":")).encode("utf-8"),
        )
    except Exception:
        # A training-data disk problem must never turn a valid student
        # verification into a failed request.
        logger.exception("Could not persist training capture %s.", capture_id)
        for path_name in (
            locals().get("image_path"),
            locals().get("metadata_path"),
        ):
            if path_name is not None:
                try:
                    path_name.unlink(missing_ok=True)
                except OSError:
                    pass
        return None
    return capture_id


def _atomic_write(path: Path, content: bytes) -> None:
    """Write a file privately, then rename it into the capture directory."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
