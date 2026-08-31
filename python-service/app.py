"""
Flask microservice exposing the passport photo verification engine over HTTP.
PHP calls this service (multipart/form-data) with the image + enabled criteria.

Run:
    pip install -r requirements.txt
    python app.py
Service listens on http://127.0.0.1:5001
"""

import io
import json
import logging
import os
import threading
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
from service_auth import register_service_auth, validate_service_auth_config
from verify import (
    run_checks,
    CHECK_LABELS,
    ImageValidationError,
    _detect_faces,
    _int_env,
    get_tie_detector,
    load_validated_image,
    perception_health,
)
from training_capture import capture_inference_example

logger = logging.getLogger(__name__)


def _configure_logging():
    """Configure root logging once, unless the host already did it."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_configure_logging()


try:
    from rembg import new_session as rembg_new_session, remove as rembg_remove
    REMBG_AVAILABLE = True
except Exception:
    rembg_new_session = None
    rembg_remove = None
    REMBG_AVAILABLE = False

# Lazy-loaded rembg session for background editing
REMBG_SESSION = None
_REMBG_INIT_ATTEMPTED = False
_REMBG_LOCK = threading.Lock()

app = Flask(__name__)

# Reject forged Host headers before routing. The default is appropriate for the
# documented loopback deployment; a private reverse-proxy address can be
# supplied explicitly by the supervisor as TRUSTED_HOSTS.
_trusted_hosts = [
    host.strip()
    for host in os.environ.get("TRUSTED_HOSTS", "127.0.0.1,localhost").split(",")
    if host.strip()
]
app.config["TRUSTED_HOSTS"] = _trusted_hosts or ["127.0.0.1", "localhost"]

MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 12, minimum=1, maximum=100)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Upper bound for the JSON form fields, so a caller cannot send megabytes of
# criteria/params text alongside a small image.
MAX_JSON_FIELD_CHARS = _int_env(
    "MAX_JSON_FIELD_CHARS", 20_000, minimum=1_000, maximum=1_000_000
)

# Cap non-file multipart fields at the Werkzeug layer as well, so an oversized
# criteria/params body is rejected before it is buffered into memory.
# https://flask.palletsprojects.com/en/stable/config/ (Flask >= 3.1)
app.config["MAX_FORM_MEMORY_SIZE"] = max(500_000, MAX_JSON_FIELD_CHARS * 4)

# Largest output the editor will render, independent of the input size.
MAX_EDIT_DIMENSION = _int_env("MAX_EDIT_DIMENSION", 6_000, minimum=100, maximum=20_000)
MAX_EDIT_PIXELS = _int_env(
    "MAX_EDIT_PIXELS", 40_000_000, minimum=1_000_000, maximum=200_000_000
)

# Endpoints reachable without the shared service credential. Keep this
# minimal: /health is the liveness probe for process supervisors.
PUBLIC_ENDPOINTS = frozenset({"health"})
register_service_auth(app, PUBLIC_ENDPOINTS)

# ---------------------------------------------------------------------------
# CPU thread budget
# ---------------------------------------------------------------------------
# Each sync Gunicorn worker runs its own Torch/ONNX inference. Left at the
# default, every worker claims every core, so N workers oversubscribe the CPU
# by a factor of N and all of them get slower under concurrent load. Divide the
# cores between workers instead.
def _configure_inference_threads():
    try:
        cores = os.cpu_count() or 1
        workers = _int_env("GUNICORN_WORKERS", 2, minimum=1, maximum=256)
        default_threads = max(1, cores // max(1, workers))
        threads = _int_env(
            "TORCH_NUM_THREADS", default_threads, minimum=1, maximum=max(1, cores)
        )
        # Also constrain the native BLAS/OpenMP pools used by numpy/OpenCV.
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        cv2.setNumThreads(threads)
        import torch

        torch.set_num_threads(threads)
        logger.info(
            "Inference thread budget: %d thread(s) per worker (%d cores, %d workers).",
            threads, cores, workers,
        )
    except Exception:
        logger.exception("Could not configure the inference thread budget.")


_configure_inference_threads()


# ---------------------------------------------------------------------------
# Caller deadlines
# ---------------------------------------------------------------------------
# Verification is CPU-bound and can take tens of seconds under concurrency. If
# PHP has already given up on a queued request, running the models anyway just
# steals CPU from requests whose caller is still waiting. PHP sends the instant
# it stops waiting; a worker that picks the request up after that point drops it
# before doing any inference.
DEADLINE_HEADER = "X-Request-Deadline"


def _deadline_exceeded():
    """True when the caller's stated deadline has already passed."""
    raw = request.headers.get(DEADLINE_HEADER, "").strip()
    if not raw:
        return False
    try:
        deadline_ms = float(raw)
    except (TypeError, ValueError):
        return False
    if deadline_ms <= 0:
        return False
    return (time.time() * 1000.0) > deadline_ms


def _abandoned_response(endpoint):
    logger.warning(
        "Dropping %s: the caller's deadline had already passed when this worker "
        "picked the request up (service is over capacity).", endpoint,
    )
    return jsonify({
        "error": "The verification service is busy. Please try again.",
        "code": "SERVICE_BUSY",
    }), 503

# CORS is opt-in. The service is designed for server-to-server calls from the
# PHP portal, which do not need CORS at all. Browsers only need it if the
# portal calls this service directly from client-side JavaScript.
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("PORTAL_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _cors_origins:
    CORS(
        app,
        origins=_cors_origins,
        allow_headers=["Content-Type", "X-Service-Token"],
        methods=["GET", "POST", "OPTIONS"],
    )
    logger.info("CORS enabled for origins: %s", ", ".join(_cors_origins))


# ---------- JSON error handling ----------
# Without these, Flask returns HTML for 413/404/500, which breaks the PHP
# client's json_decode() and surfaces as "Invalid response from verification
# service."

@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    payload = {
        "error": exc.description,
        "code": exc.name.upper().replace(" ", "_"),
    }
    if exc.code == 413:
        payload["error"] = f"Uploaded file is too large. Maximum size is {MAX_UPLOAD_MB} MB."
    return jsonify(payload), exc.code


@app.errorhandler(Exception)
def handle_unexpected_exception(exc):
    logger.exception("Unhandled error while serving %s", request.path)
    return jsonify({
        "error": "Internal server error.",
        "code": "INTERNAL_ERROR",
    }), 500


def get_rembg_session():
    """Load the background-cutout session once, only when editing needs it.

    Default is ``u2netp`` (the portable U2-Net). Full ``u2net`` is ~176 MB and
    was taking 12–21 s per photograph on CPU — longer than verification
    itself once tie detection is off. ``u2netp`` is the same architecture at
    a fraction of the size; override with REMBG_MODEL if a site needs the
    large model after measuring cutout quality.
    """
    global REMBG_SESSION, _REMBG_INIT_ATTEMPTED
    if not REMBG_AVAILABLE:
        return None
    with _REMBG_LOCK:
        if not _REMBG_INIT_ATTEMPTED:
            _REMBG_INIT_ATTEMPTED = True
            model = _rembg_model_name()
            try:
                REMBG_SESSION = rembg_new_session(model)
                logger.info("rembg session loaded (%s).", model)
            except Exception:
                logger.exception("rembg session (%s) could not be initialised.", model)
                REMBG_SESSION = None
    return REMBG_SESSION


_ALLOWED_REMBG_MODELS = frozenset({
    "u2netp",
    "u2net",
    "u2net_human_seg",
    "silueta",
    "isnet-general-use",
})


def _rembg_model_name():
    raw = os.environ.get("REMBG_MODEL", "u2netp").strip().lower()
    if raw not in _ALLOWED_REMBG_MODELS:
        logger.warning("Invalid REMBG_MODEL=%r; using u2netp.", raw)
        return "u2netp"
    return raw


def _image_for_cutout_inference(pil_img_rgb):
    """Downscale a photo before rembg, returning (inference_image, orig_size_or_None).

    rembg's cost scales with pixel count. Passport output is a few hundred
    pixels on a side; running U2-Net on a 3000px phone original is wasted.
    ``orig_size`` is None when no resize happened.
    """
    max_side = _int_env("EDIT_INFERENCE_MAX_SIDE", 768, minimum=256, maximum=4000)
    width, height = pil_img_rgb.size
    longest = max(width, height)
    if longest <= max_side:
        return pil_img_rgb, None
    scale = max_side / float(longest)
    small = pil_img_rgb.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )
    return small, (width, height)

def _warm_all_models():
    """Load both lazily-initialized ML models once so first requests are fast."""
    result = {}
    try:
        get_tie_detector()
        result["tie_detector_loaded"] = True
    except Exception as exc:  # never let warm-up crash the caller
        logger.exception("Tie detector could not be loaded during warm-up.")
        result["tie_detector_loaded"] = False
        # Model-loader errors can contain filesystem paths or dependency
        # details. Keep those in server logs, not in an API response.
        result["tie_detector_error"] = type(exc).__name__
    result["rembg_session_loaded"] = get_rembg_session() is not None
    return result

@app.route("/warmup", methods=["GET", "POST"])
def warmup():
    """Pre-load lazy ML models. Cheap/no-op when already warm."""
    return jsonify({"status": "ok", **_warm_all_models()}
)


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness probe: reports whether the ML models are actually usable.

    ``/health`` only proves the process is alive. This endpoint proves the
    service can serve a verification, so orchestrators and deploy scripts can
    gate traffic on real readiness instead of process liveness.
    """
    status = _warm_all_models()
    tie_required = os.environ.get("REQUIRE_TIE_MODEL_READY", "1").strip().lower() in (
        "1", "true", "yes",
    )
    rembg_required = os.environ.get("REQUIRE_REMBG_READY", "0").strip().lower() in (
        "1", "true", "yes",
    )
    landmarks_required = os.environ.get("REQUIRE_LANDMARKS_READY", "1").strip().lower() in (
        "1", "true", "yes",
    )

    perception = perception_health()
    is_ready = True
    if tie_required and not status.get("tie_detector_loaded"):
        is_ready = False
    if rembg_required and not status.get("rembg_session_loaded"):
        is_ready = False
    # Without landmarks the head-pose and eyes-open criteria cannot pass, so a
    # worker in that state must be pulled out of rotation rather than quietly
    # rejecting every applicant.
    if landmarks_required and not perception["landmarks_available"]:
        is_ready = False

    return jsonify({
        "status": "ready" if is_ready else "not_ready",
        **status,
        "perception": perception,
    }), (200 if is_ready else 503)

def get_subject_cutout(pil_img):
    """
    Isolates the subject/person completely using U2-Net deep learning portrait segmentation (rembg)
    or OpenCV GrabCut/FloodFill fallback, returning a PIL Image in RGBA mode with a transparent background.
    """
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img_rgb = pil_img.convert("RGB")

    cutout_img = None
    rembg_session = get_rembg_session()
    if rembg_session is not None:
        try:
            inference_img, orig_size = _image_for_cutout_inference(pil_img_rgb)
            cutout_img = rembg_remove(inference_img, session=rembg_session)
            if orig_size is not None and cutout_img is not None:
                cutout_img = cutout_img.resize(orig_size, Image.BILINEAR)
        except Exception as e:
            logger.exception("rembg_remove failed: %s", e)

    if cutout_img is not None:
        arr_rgba = np.array(cutout_img)
        if arr_rgba.ndim == 3 and arr_rgba.shape[2] == 4:
            alpha = arr_rgba[:, :, 3].copy()
            # Aggressively clean up low-alpha background noise and fringe artifacts
            alpha[alpha < 25] = 0
            valid_mask = alpha >= 25
            alpha[valid_mask] = np.clip((alpha[valid_mask].astype(np.float32) - 25.0) * (255.0 / 230.0), 0, 255).astype(np.uint8)
            arr_rgba[:, :, 3] = alpha
            return Image.fromarray(arr_rgba, mode="RGBA")

    arr = np.array(pil_img_rgb)
    h, w = arr.shape[:2]
    if h < 10 or w < 10:
        return pil_img.convert("RGBA")

    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    try:
        border_w = max(2, int(w * 0.05))
        border_h = max(2, int(h * 0.05))

        top_strip = bgr[0:border_h, :]
        bot_strip = bgr[h - border_h:h, :]
        left_strip = bgr[:, 0:border_w]
        right_strip = bgr[:, w - border_w:w]

        bg_samples = np.concatenate([
            top_strip.reshape(-1, 3),
            bot_strip.reshape(-1, 3),
            left_strip.reshape(-1, 3),
            right_strip.reshape(-1, 3)
        ], axis=0)

        mask = np.zeros(bgr.shape[:2], np.uint8)
        mask[:] = cv2.GC_PR_BGD

        border = max(2, int(min(w, h) * 0.02))
        mask[0:border, :] = cv2.GC_BGD
        mask[:, 0:border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        rect = (border, border, w - 2 * border, h - border)

        faces = _detect_faces(bgr)
        if faces:
            f = faces[0]
            fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]

            body_x1 = max(0, fx - int(fw * 0.8))
            body_y1 = max(0, fy - int(fh * 0.4))
            body_x2 = min(w, fx + int(fw * 1.8))
            body_y2 = min(h, fy + int(fh * 3.8))
            mask[body_y1:body_y2, body_x1:body_x2] = cv2.GC_PR_FGD

            core_x1 = max(0, fx + int(fw * 0.15))
            core_y1 = max(0, fy + int(fh * 0.15))
            core_x2 = min(w, fx + int(fw * 0.85))
            core_y2 = min(h, fy + int(fh * 0.85))
            mask[core_y1:core_y2, core_x1:core_x2] = cv2.GC_FGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        bgr_copy = bgr.copy()
        
        cv2.floodFill(bgr_copy, ff_mask, (0, 0), (0, 0, 0), (25, 25, 25), (25, 25, 25), cv2.FLOODFILL_FIXED_RANGE)
        cv2.floodFill(bgr_copy, ff_mask, (w - 1, 0), (0, 0, 0), (25, 25, 25), (25, 25, 25), cv2.FLOODFILL_FIXED_RANGE)
        
        flood_bg_mask = (ff_mask[1:-1, 1:-1] > 0).astype(np.float32)

        try:
            cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            grabcut_bg_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 1.0, 0.0)
        except Exception:
            grabcut_bg_mask = flood_bg_mask

        final_bg_mask = np.minimum(grabcut_bg_mask, flood_bg_mask)

        fg_alpha = ((1.0 - final_bg_mask) * 255).astype(np.uint8)
        arr_rgba = np.dstack((arr, fg_alpha))
        return Image.fromarray(arr_rgba, mode="RGBA")
    except Exception:
        return pil_img.convert("RGBA")


def _create_dynamic_backdrop(width, height, target_rgb, center_x=None, center_y=None):
    """
    Generate a studio backdrop. For pure white target (#ffffff), returns solid pure 255 white
    without any vignette darkening at the borders to meet strict biometric passport standard.
    """
    R, G, B = target_rgb
    if R >= 240 and G >= 240 and B >= 240:
        # High-Key Biometric Studio Pure White (#FFFFFF) — 100% uniform solid white
        bg = np.full((height, width, 3), 255.0, dtype=np.float32)
        return bg

    cx = center_x if center_x is not None else width * 0.5
    cy = center_y if center_y is not None else height * 0.38

    Y, X = np.ogrid[:height, :width]
    diag = np.sqrt(width**2 + height**2)
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2) / max(1.0, diag)

    # Dynamic Studio Key Light Spotlight (+12% center hotspot, -12% perimeter)
    lum = 1.12 - 0.24 * (dist ** 1.2)

    bg = np.zeros((height, width, 3), dtype=np.float32)
    bg[:, :, 0] = np.clip(R * lum, 0, 255)
    bg[:, :, 1] = np.clip(G * lum, 0, 255)
    bg[:, :, 2] = np.clip(B * lum, 0, 255)
    return bg


def replace_background_color(pil_img, hex_color):
    """
    Dynamic AI Background Replacement & Studio Lighting Alignment:
    1. Isolates subject with smooth alpha matting.
    2. Generates dynamic studio backdrop aligned behind the subject.
    3. Harmonizes edge fringe with subtle light wrap for seamless integration.
    """
    if not hex_color or str(hex_color).strip().lower() in ("", "none", "keep", "original", "transparent"):
        return pil_img

    hex_clean = str(hex_color).lstrip('#').strip()
    if len(hex_clean) != 6:
        return pil_img

    try:
        target_rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return pil_img

    pil_img = ImageOps.exif_transpose(pil_img)
    cutout_rgba = get_subject_cutout(pil_img)
    cutout_arr = np.array(cutout_rgba)
    if cutout_arr.ndim != 3 or cutout_arr.shape[2] != 4:
        return pil_img

    w, h = cutout_rgba.size
    alpha = cutout_arr[:, :, 3].astype(np.float32) / 255.0

    is_white_bg = (target_rgb[0] >= 240 and target_rgb[1] >= 240 and target_rgb[2] >= 240)

    # Locate subject head center for dynamic background alignment. Pure white
    # (#ffffff, what the portal always requests) ignores the center — the
    # backdrop is uniform — so a second MediaPipe pass here is wasted time.
    center_x = w * 0.5
    center_y = h * 0.38
    if not is_white_bg:
        try:
            bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            faces = _detect_faces(bgr)
            if faces:
                f = faces[0]
                center_x = f["x"] + f["w"] * 0.5
                center_y = f["y"] + f["h"] * 0.45
        except Exception:
            pass

    # Dynamic studio backdrop aligned with subject's position
    bg_layer = _create_dynamic_backdrop(w, h, target_rgb, center_x, center_y)

    fg_rgb = cutout_arr[:, :, :3].astype(np.float32)

    if is_white_bg:
        # Eliminate low-confidence background noise
        alpha[alpha < 0.10] = 0.0

    # Dynamic Light Wrap / Edge Harmonization on boundary transition zone
    fringe = (alpha > 0.02) & (alpha < 0.92)
    if np.any(fringe):
        if is_white_bg:
            # Defringe transition edge so dark background remnants don't leave dark halos
            for c in range(3):
                fg_rgb[fringe, c] = np.maximum(fg_rgb[fringe, c], bg_layer[fringe, c] * (1.0 - alpha[fringe]))
        else:
            wrap_factor = (1.0 - alpha[fringe]) * 0.18
            for c in range(3):
                fg_rgb[fringe, c] = fg_rgb[fringe, c] * (1.0 - wrap_factor) + bg_layer[fringe, c] * wrap_factor

    # Alpha composite
    comp = fg_rgb * alpha[:, :, np.newaxis] + bg_layer * (1.0 - alpha[:, :, np.newaxis])
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return Image.fromarray(comp, mode="RGB")


def _apply_gamma_brightness(pil_img, brightness_value):
    """
    Apply brightness adjustment with gamma curve in LAB luminance space.
    """
    if brightness_value == 0:
        return pil_img

    b = float(brightness_value)
    gamma = 1.0 / (1.0 + b / 80.0) if b > 0 else 1.0 + abs(b) / 80.0
    gamma = max(0.3, min(3.0, gamma))

    arr = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    L_gamma = np.power(L, gamma)
    lab[:, :, 0] = np.clip(L_gamma * 255.0, 0, 255)

    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result, mode="RGB")


def _apply_unsharp_mask(pil_img, sharpness_value):
    """
    Apply responsive photographic high-pass unsharp mask on luminance.
    """
    s = float(sharpness_value)
    if s <= 1.0:
        return pil_img

    strength = min(2.5, (s - 1.0) * 1.5)
    arr = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]

    h, w = L.shape[:2]
    radius = max(1.2, min(w, h) / 450.0)
    ksize = int(radius * 3.0) | 1
    ksize = max(3, ksize)

    L_blurred = cv2.GaussianBlur(L, (ksize, ksize), radius)
    detail = L - L_blurred
    detail[np.abs(detail) < 0.5] = 0.0

    lab[:, :, 0] = np.clip(L + strength * detail, 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result, mode="RGB")


def process_photo_edits(image_bytes, brightness=0.0, width=None, height=None, sharpness=1.0, bg_color=None):
    """
    Applies background replacement (with dynamic studio lighting & light wrap),
    gamma brightness, photographic sharpness, and optional resizing.
    """
    pil_img = load_validated_image(image_bytes)

    # 1. Background Color Replacement with dynamic studio lighting & light wrap
    if bg_color and str(bg_color).strip():
        pil_img = replace_background_color(pil_img, str(bg_color).strip())

    # 2. Brightness adjustment
    try:
        b_val = float(brightness)
        if b_val != 0.0 and -100 <= b_val <= 100:
            pil_img = _apply_gamma_brightness(pil_img, b_val)
    except (ValueError, TypeError):
        pass

    # 3. Sharpness adjustment
    try:
        s_val = float(sharpness)
        if s_val > 1.0:
            pil_img = _apply_unsharp_mask(pil_img, s_val)
    except (ValueError, TypeError):
        pass

    # 4. Resolution / Resizing (bounded: an unbounded resize is a trivial
    #    memory-exhaustion vector, e.g. width=100000&height=100000)
    if width and height:
        try:
            w = int(width)
            h = int(height)
        except (ValueError, TypeError):
            raise ImageValidationError("Resize dimensions must be integers.")
        if w <= 0 or h <= 0:
            raise ImageValidationError("Resize dimensions must be positive.")
        if w > MAX_EDIT_DIMENSION or h > MAX_EDIT_DIMENSION:
            raise ImageValidationError(
                f"Resize dimensions must not exceed {MAX_EDIT_DIMENSION} pixels per side."
            )
        if w * h > MAX_EDIT_PIXELS:
            raise ImageValidationError(
                f"Requested output exceeds the maximum of {MAX_EDIT_PIXELS} pixels."
            )
        pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)

    out_buf = io.BytesIO()
    pil_img.save(out_buf, format="JPEG", quality=96, subsampling=0)
    return out_buf.getvalue()


@app.route("/health", methods=["GET"])
def health():
    perception = perception_health()
    return jsonify({
        # A worker that cannot produce landmarks still serves requests, but it
        # rejects valid photographs, so liveness alone is misleading.
        "status": "ok" if perception["landmarks_available"] else "degraded",
        "service": "passport-verification",
        "checks_available": list(CHECK_LABELS.keys()),
        "perception": perception,
    })


def _parse_json_object(raw, field_name, *, default_on_error=None):
    """Parse a form field that must contain a JSON object.

    Returns ``(value, error_response)``. Anything that is valid JSON but not an
    object (``[]``, ``null``, ``"x"``, ``3``) is rejected, because downstream
    code calls ``.get()`` on it and would otherwise raise a 500.
    """
    if raw is None:
        return {}, None
    if len(raw) > MAX_JSON_FIELD_CHARS:
        return None, (jsonify({
            "error": f"'{field_name}' field is too large.",
            "code": "FIELD_TOO_LARGE",
        }), 400)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        if default_on_error is not None:
            return dict(default_on_error), None
        return None, (jsonify({
            "error": f"Invalid '{field_name}' JSON.",
            "code": "INVALID_JSON",
        }), 400)

    if not isinstance(value, dict):
        if default_on_error is not None:
            return dict(default_on_error), None
        return None, (jsonify({
            "error": f"'{field_name}' must be a JSON object.",
            "code": "INVALID_JSON_TYPE",
        }), 400)
    return value, None


@app.route("/verify", methods=["POST"])
def verify():
    if _deadline_exceeded():
        return _abandoned_response("/verify")

    if "photo" not in request.files:
        return jsonify({"error": "No photo file provided (field name must be 'photo')."}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file."}), 400

    enabled_criteria, error = _parse_json_object(
        request.form.get("criteria", "{}"), "criteria"
    )
    if error:
        return error

    # Params are advisory: a malformed value falls back to defaults rather than
    # rejecting an otherwise valid student submission.
    params, error = _parse_json_object(
        request.form.get("params", "{}"), "params", default_on_error={}
    )
    if error:
        return error

    # Validate and normalize background parameters
    level = str(params.get("background_strictness", "standard")).strip().lower()
    params["background_strictness"] = (
        level if level in ("strict", "standard", "relaxed", "accept_all") else "standard"
    )

    if "background_near_white_acceptance" in params:
        switch = str(params["background_near_white_acceptance"]).strip().lower()
        if switch not in ("auto", "1", "0"):
            switch = "auto"
        params["background_near_white_acceptance"] = switch
    elif "bg_near_white_enabled" in params:
        legacy = str(params["bg_near_white_enabled"]).strip().lower()
        params["background_near_white_acceptance"] = (
            legacy if legacy in ("auto", "1", "0") else "auto"
        )
        del params["bg_near_white_enabled"]

    for key in [k for k in params if k.startswith("bg_")]:
        del params[key]

    result = run_checks(image_bytes, enabled_criteria, params)
    if (
        "error" not in result
        and request.headers.get("X-Training-Consent", "").strip().lower() in (
            "1", "true", "yes",
        )
    ):
        capture_inference_example(
            image_bytes,
            result,
            identity_id=request.headers.get("X-Applicant-Identity"),
            attire_policy=params.get("attire_policy"),
            file_extension={
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }.get(file.mimetype, ".jpg"),
        )
    status_code = 200 if "error" not in result else 422
    return jsonify(result), status_code


@app.route("/edit-photo", methods=["POST"])
def edit_photo():
    if _deadline_exceeded():
        return _abandoned_response("/edit-photo")

    if "photo" not in request.files:
        return jsonify({"error": "No photo file provided."}), 400

    file = request.files["photo"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty photo file."}), 400

    get_cutout = request.form.get("get_cutout", "false").lower() in ("true", "1")
    if get_cutout:
        try:
            pil_img = load_validated_image(image_bytes)
            cutout_img = get_subject_cutout(pil_img)
            out_buf = io.BytesIO()
            cutout_img.save(out_buf, format="PNG")
            return send_file(io.BytesIO(out_buf.getvalue()), mimetype="image/png")
        except ImageValidationError as exc:
            return jsonify({"error": str(exc), "code": "INVALID_IMAGE"}), 400
        except Exception:
            logger.exception("Failed to generate subject cutout.")
            return jsonify({
                "error": "Failed to generate cutout.",
                "code": "CUTOUT_FAILED",
            }), 500

    brightness = request.form.get("brightness", 0.0)
    width = request.form.get("width", None)
    height = request.form.get("height", None)
    sharpness = request.form.get("sharpness", 1.0)
    bg_color = request.form.get("bg_color", None)

    try:
        edited_bytes = process_photo_edits(
            image_bytes,
            brightness=brightness,
            width=width,
            height=height,
            sharpness=sharpness,
            bg_color=bg_color
        )
        return send_file(io.BytesIO(edited_bytes), mimetype="image/jpeg")
    except ImageValidationError as exc:
        return jsonify({"error": str(exc), "code": "INVALID_IMAGE"}), 400
    except Exception:
        logger.exception("Failed to process photo edit.")
        return jsonify({
            "error": "Failed to process photo edit.",
            "code": "EDIT_FAILED",
        }), 500


def warm_models_in_background():
    """Start model warm-up without blocking server start-up.

    Called from the Gunicorn ``post_worker_init`` hook in gunicorn.conf.py and
    from the development entrypoint below.
    """
    threading.Thread(target=_warm_all_models, name="model-warmup", daemon=True).start()


if __name__ == "__main__":
    # Development convenience only. Production must run a real WSGI server:
    #   gunicorn -c gunicorn.conf.py app:app
    # https://flask.palletsprojects.com/en/stable/deploying/
    validate_service_auth_config()
    port = _int_env("PORT", 5001, minimum=1, maximum=65535)
    logger.warning(
        "Starting Flask development server on 127.0.0.1:%s. "
        "Do NOT use this in production; run gunicorn -c gunicorn.conf.py app:app",
        port,
    )
    warm_models_in_background()
    app.run(host="127.0.0.1", port=port, debug=False)
