"""Gunicorn configuration for the passport verification service.

Run with:
    gunicorn -c gunicorn.conf.py app:app

Flask's built-in server must not be used in production:
https://flask.palletsprojects.com/en/stable/deploying/
Setting reference: https://docs.gunicorn.org/en/stable/settings.html
"""

import multiprocessing
import os


def _int_env(name, default, minimum=1, maximum=None):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    if value < minimum:
        return default
    if maximum is not None and value > maximum:
        return maximum
    return value


# Bind to loopback by default: only the PHP application should reach this
# service. Override with GUNICORN_BIND only when PHP runs on another host,
# and firewall the port to that host.
bind = os.environ.get("GUNICORN_BIND", "127.0.0.1:5001")

# Each worker loads Faster R-CNN + U2-Net + MediaPipe into its own memory
# space; these models are NOT shared between processes. Measure resident
# memory per worker on the target server before increasing this.
workers = _int_env("GUNICORN_WORKERS", 2, minimum=1, maximum=multiprocessing.cpu_count() * 2)

# Synchronous workers: CPU-bound inference gains nothing from async workers.
worker_class = "sync"

# Threads stay at 1. Torch inference is CPU-heavy and the models are shared
# mutable objects within a process.
threads = 1

# Model inference on CPU can take several seconds on a cold cache or a large
# image. The default 30s kills legitimate requests.
timeout = _int_env("GUNICORN_TIMEOUT", 120, minimum=30)
graceful_timeout = _int_env("GUNICORN_GRACEFUL_TIMEOUT", 30, minimum=5)

# Keep-alive is pointless for server-to-server calls from PHP cURL.
keepalive = 2

# Recycle workers periodically to contain any slow memory growth in the
# native imaging/ML stack. Jitter avoids all workers restarting together.
max_requests = _int_env("GUNICORN_MAX_REQUESTS", 200, minimum=0)
max_requests_jitter = _int_env("GUNICORN_MAX_REQUESTS_JITTER", 50, minimum=0)

# preload_app is deliberately OFF. Loading Torch/ONNX models in the master
# process and then forking is a known source of native-library and thread
# state issues. Each worker loads its own models via post_worker_init below.
preload_app = False

accesslog = os.environ.get("GUNICORN_ACCESS_LOG", "-")
errorlog = os.environ.get("GUNICORN_ERROR_LOG", "-")
loglevel = os.environ.get("GUNICORN_LOG_LEVEL", "info")

# Combined log format: method/path/status only. Do not add headers — the
# Authorization bearer credential must never appear in access logs.
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Bound request line/header sizes.
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

proc_name = "passport-verification"


def on_starting(server):
    """Fail the master process before workers import the vision stack."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from service_auth import validate_service_auth_config

    validate_service_auth_config()


def post_worker_init(worker):
    """Warm the ML models in each worker so the first student upload is fast."""
    from service_auth import validate_service_auth_config

    # Must not be caught: a worker without a valid secret must not serve.
    validate_service_auth_config()
    try:
        from app import warm_models_in_background

        warm_models_in_background()
    except Exception:
        worker.log.exception("Model warm-up could not be started.")
