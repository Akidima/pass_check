"""Shared helper for tests that call the Flask service over the test client.

Every endpoint except ``/health`` requires the Authorization bearer
credential that the PHP portal sends. Tests therefore have to authenticate
exactly like the real caller does, otherwise they would only ever exercise
the 401 path.
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Longer than service_auth.MIN_SECRET_LENGTH. Not a production secret.
TEST_SERVICE_TOKEN = "test-only-service-token-do-not-use-in-prod"


def authenticated_client(flask_app, *, testing=True):
    """Return a test client that authenticates like the PHP portal."""
    os.environ["PORTAL_SHARED_SECRET"] = TEST_SERVICE_TOKEN
    flask_app.config["TESTING"] = testing
    client = flask_app.test_client()
    client.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {TEST_SERVICE_TOKEN}"
    return client


def unauthenticated_client(flask_app):
    """Return a test client that sends no service credential."""
    os.environ["PORTAL_SHARED_SECRET"] = TEST_SERVICE_TOKEN
    flask_app.config["TESTING"] = True
    return flask_app.test_client()
