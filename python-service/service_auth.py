"""Server-to-server authentication for the verification service.

The PHP portal is the only intended caller. It sends the shared secret from
its process environment; this module rejects the request before any image
processing if that credential is missing or wrong.

This module must stay free of OpenCV, MediaPipe, and model imports so
Gunicorn can validate configuration in the master process before workers
load the vision stack.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from hashlib import sha256
from typing import Iterable, Optional, Sequence

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

# 32 bytes of entropy encoded as hex/urlsafe is at least 43 characters.
# Reject shorter configured values so a typo or placeholder cannot ship.
MIN_SECRET_LENGTH = 32

# Header the PHP portal should send.
AUTHORIZATION_HEADER = "Authorization"
BEARER_SCHEME = "bearer"

# Legacy header still accepted so an already-deployed PHP process keeps
# working until it is updated to Authorization: Bearer.
LEGACY_TOKEN_HEADER = "X-Service-Token"

# Only this exact combination allows the process to start without a valid
# secret. Omitting either variable is treated as production.
_DEV_ENV_VALUES = frozenset({"development", "dev"})
_INSECURE_START_FLAG = "1"

# Rate-limit auth-failure logs so a scanner cannot flood disk.
# None means "never logged". Do not use 0.0 — time.monotonic() is often a
# small number after boot or in a container, which would swallow the first
# 10 seconds of failures.
_AUTH_FAIL_LOG_INTERVAL_SECONDS = 10.0
_auth_fail_lock = threading.Lock()
_auth_fail_count = 0
_auth_fail_last_log = None


class ServiceAuthConfigError(RuntimeError):
    """Raised when production authentication cannot be configured safely."""


def _app_env() -> str:
    return os.environ.get("APP_ENV", "").strip().lower()


def allows_insecure_startup() -> bool:
    """True only when an operator explicitly opts into local-dev startup.

    Accidental omission of environment variables must not disable this check.
    ``APP_ENV=development`` alone is not enough.
    """
    flag = os.environ.get("ALLOW_INSECURE_AUTH_START", "").strip()
    return flag == _INSECURE_START_FLAG and _app_env() in _DEV_ENV_VALUES


def _normalize_secret(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def configured_secrets() -> tuple[str, ...]:
    """Return the active secret and, if set, the previous secret for rotation."""
    current = _normalize_secret(os.environ.get("PORTAL_SHARED_SECRET"))
    previous = _normalize_secret(os.environ.get("PORTAL_SHARED_SECRET_PREVIOUS"))
    secrets: list[str] = []
    if current:
        secrets.append(current)
    if previous and previous != current:
        secrets.append(previous)
    return tuple(secrets)


def _usable_secrets(candidates: Sequence[str]) -> tuple[str, ...]:
    return tuple(secret for secret in candidates if len(secret) >= MIN_SECRET_LENGTH)


def validate_service_auth_config() -> None:
    """Refuse to start when production authentication is not safely configured.

    Does not log or return the secret value.
    """
    if allows_insecure_startup():
        logger.error(
            "Starting without a valid PORTAL_SHARED_SECRET because "
            "APP_ENV is development and ALLOW_INSECURE_AUTH_START=1. "
            "Authenticated endpoints will still return 401. "
            "This combination is forbidden in production."
        )
        return

    current = _normalize_secret(os.environ.get("PORTAL_SHARED_SECRET"))
    if current is None:
        raise ServiceAuthConfigError(
            "PORTAL_SHARED_SECRET is missing. The service will not start "
            "without a shared secret of at least "
            f"{MIN_SECRET_LENGTH} characters."
        )
    if len(current) < MIN_SECRET_LENGTH:
        raise ServiceAuthConfigError(
            "PORTAL_SHARED_SECRET is too short. Generate at least "
            f"{MIN_SECRET_LENGTH} characters of entropy, for example with "
            '`python -c "import secrets; print(secrets.token_urlsafe(32))"`.'
        )

    previous = _normalize_secret(os.environ.get("PORTAL_SHARED_SECRET_PREVIOUS"))
    if previous is not None and len(previous) < MIN_SECRET_LENGTH:
        logger.error(
            "PORTAL_SHARED_SECRET_PREVIOUS is too short and will be ignored."
        )


def _digest(value: str) -> bytes:
    return sha256(value.encode("utf-8")).digest()


def secrets_match(provided: str, expected_secrets: Sequence[str]) -> bool:
    """Constant-time compare against every configured secret.

    Values are hashed first so a length mismatch cannot short-circuit
    ``hmac.compare_digest``. Every candidate is always compared.
    """
    provided_digest = _digest(provided)
    matched = False
    for expected in expected_secrets:
        matched = hmac.compare_digest(provided_digest, _digest(expected)) or matched
    if not expected_secrets:
        hmac.compare_digest(provided_digest, _digest(""))
    return matched


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Return the Bearer credential, or None if the header is not Bearer auth."""
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0], parts[1].strip()
    if scheme.lower() != BEARER_SCHEME:
        return None
    return token or None


def extract_presented_secret() -> Optional[str]:
    """Read the caller credential from the preferred or legacy header.

    If ``Authorization`` is present it must be a valid Bearer token; the
    legacy header is not consulted in that case. This avoids accepting a
    leftover ``X-Service-Token`` after a proxy injects a broken
    Authorization header.
    """
    if AUTHORIZATION_HEADER in request.headers:
        return extract_bearer_token(request.headers.get(AUTHORIZATION_HEADER))

    legacy = request.headers.get(LEGACY_TOKEN_HEADER)
    if legacy is None:
        return None
    token = legacy.strip()
    return token or None


def _log_auth_failure(path: str) -> None:
    """Log an auth failure without credentials, headers, or request bodies."""
    global _auth_fail_count, _auth_fail_last_log
    now = time.monotonic()
    with _auth_fail_lock:
        _auth_fail_count += 1
        if (
            _auth_fail_last_log is not None
            and now - _auth_fail_last_log < _AUTH_FAIL_LOG_INTERVAL_SECONDS
        ):
            return
        count = _auth_fail_count
        _auth_fail_count = 0
        _auth_fail_last_log = now
    logger.warning(
        "Unauthorized request rejected for %s (%d failure(s) since last log)",
        path,
        count,
    )


def reset_auth_failure_log_state_for_tests() -> None:
    global _auth_fail_count, _auth_fail_last_log
    with _auth_fail_lock:
        _auth_fail_count = 0
        _auth_fail_last_log = None


def unauthorized_response():
    """Generic 401. Must not mention the secret, header names, or config."""
    return jsonify({"error": "Unauthorized"}), 401


def authenticate_request(public_endpoints: Iterable[str]):
    """Flask ``before_request`` hook. Returns a 401 response or None."""
    if request.endpoint in frozenset(public_endpoints):
        return None
    # CORS preflight never carries credentials.
    if request.method == "OPTIONS":
        return None

    expected = _usable_secrets(configured_secrets())
    if not expected:
        logger.error(
            "Refusing request: PORTAL_SHARED_SECRET is missing or too short."
        )
        _log_auth_failure(request.path)
        return unauthorized_response()

    presented = extract_presented_secret()
    if presented is None or not secrets_match(presented, expected):
        _log_auth_failure(request.path)
        return unauthorized_response()
    return None


def register_service_auth(app: Flask, public_endpoints: Iterable[str]) -> None:
    """Install the authentication guard on ``app`` before any view runs."""
    public = frozenset(public_endpoints)

    @app.before_request
    def require_service_auth():
        return authenticate_request(public)
