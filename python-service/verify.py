"""
Passport/ID Photo Verification Engine
--------------------------------------
Performs computer-vision checks that PHP cannot do natively:
 - face count / single person detection
 - face size & centering (passport framing)
 - eyewear (glasses) detection
 - background color uniformity / "strictly white" check
 - tie / formal-neckwear heuristic detection
 - resolution & aspect-ratio / passport-size check
 - blur (sharpness) detection
 - brightness / exposure check
 - head tilt / pose check

Each check returns a dict: {passed: bool, message: str, meta: {...}}
The main entrypoint `run_checks(image_bytes, enabled_criteria, params)` runs
only the checks the admin has toggled on, and returns an aggregate result.
"""

import io
import logging
import os
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

from official_tie import (
    classify_neckwear_type,
    decide_require_tie,
    load_official_tie_policy,
    measure_appearance,
    match_official_appearance,
)
from tie_visibility import UpperBodyVisibilityEstimator
from tie_detector import (
    TieDetection,
    TieModelPolicy,
    get_tie_detector,
    validate_tie_detection,
)

logger = logging.getLogger(__name__)
# Historical alias used throughout the tie-detection helpers below.
_logger = logger

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    MEDIAPIPE_AVAILABLE = False

# Fallback Haar cascade for face detection when mediapipe isn't installed
_HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------------------------------------------------------
# MediaPipe runtime health
# ---------------------------------------------------------------------------
# MEDIAPIPE_AVAILABLE records whether the library could be imported. It must
# never be mutated at runtime: a single transient graph/GPU-context error would
# otherwise disable landmarks for the entire remaining life of the worker, and
# because every enabled criterion is mandatory, head_pose and eyes_open would
# then reject every following upload while /health still reported "ok".
#
# Runtime errors instead open a short, self-healing cooldown. During the
# cooldown we skip MediaPipe (so a genuinely broken install does not pay the
# initialization cost on every request); afterwards we retry automatically.
_MEDIAPIPE_COOLDOWN_SECONDS = 30.0
_MEDIAPIPE_LOCK = threading.Lock()
_MEDIAPIPE_COOLDOWN_UNTIL = 0.0
_MEDIAPIPE_FAILURES = 0
_MEDIAPIPE_LAST_ERROR = None

# Graph construction is the expensive part of MediaPipe on CPU (GL context +
# TFLite delegate). Creating FaceDetection and FaceMesh on every photograph
# was adding several seconds before any check ran, and head_pose + eyes_open
# each built a fresh FaceMesh. Sessions are reused for the life of the worker
# and guarded by the same lock (the graphs are not thread-safe).
_FACE_DETECTOR = None
_FACE_MESH = None
_LANDMARK_CACHE_ID = None
_LANDMARK_CACHE_VAL = None


def _mediapipe_usable():
    """True when MediaPipe is importable and not inside a failure cooldown."""
    if not MEDIAPIPE_AVAILABLE:
        return False
    with _MEDIAPIPE_LOCK:
        return time.monotonic() >= _MEDIAPIPE_COOLDOWN_UNTIL


def _note_mediapipe_failure(exc):
    """Open a bounded cooldown after a MediaPipe runtime error."""
    global _MEDIAPIPE_COOLDOWN_UNTIL, _MEDIAPIPE_FAILURES, _MEDIAPIPE_LAST_ERROR
    with _MEDIAPIPE_LOCK:
        _MEDIAPIPE_FAILURES += 1
        _MEDIAPIPE_LAST_ERROR = f"{type(exc).__name__}: {exc}"[:200]
        _MEDIAPIPE_COOLDOWN_UNTIL = time.monotonic() + _MEDIAPIPE_COOLDOWN_SECONDS


def _note_mediapipe_success():
    """Clear the cooldown once MediaPipe works again."""
    global _MEDIAPIPE_COOLDOWN_UNTIL
    with _MEDIAPIPE_LOCK:
        if _MEDIAPIPE_COOLDOWN_UNTIL:
            _MEDIAPIPE_COOLDOWN_UNTIL = 0.0


def perception_health():
    """Report MediaPipe health so readiness probes can see a degraded worker.

    A worker that has fallen back to Haar cascades cannot evaluate head pose or
    eyes-open, so it rejects otherwise valid photographs. That must be visible
    to operators rather than silently reducing every applicant's result.
    """
    with _MEDIAPIPE_LOCK:
        cooldown_remaining = max(0.0, _MEDIAPIPE_COOLDOWN_UNTIL - time.monotonic())
        return {
            "mediapipe_imported": bool(MEDIAPIPE_AVAILABLE),
            "landmarks_available": bool(MEDIAPIPE_AVAILABLE) and cooldown_remaining == 0.0,
            "mediapipe_failures": _MEDIAPIPE_FAILURES,
            "mediapipe_cooldown_seconds_remaining": round(cooldown_remaining, 1),
            "mediapipe_last_error": _MEDIAPIPE_LAST_ERROR,
        }


def _close_mediapipe_sessions_locked():
    """Drop cached MediaPipe graphs. Caller must hold ``_MEDIAPIPE_LOCK``."""
    global _FACE_DETECTOR, _FACE_MESH, _LANDMARK_CACHE_ID, _LANDMARK_CACHE_VAL
    for session in (_FACE_DETECTOR, _FACE_MESH):
        if session is None:
            continue
        try:
            session.close()
        except Exception:
            pass
    _FACE_DETECTOR = None
    _FACE_MESH = None
    _LANDMARK_CACHE_ID = None
    _LANDMARK_CACHE_VAL = None


def reset_perception_health():
    """Clear recorded MediaPipe failures and cached graphs.

    Tests that patch ``FaceDetection`` / ``FaceMesh`` must call this first so
    the constructor is used again rather than a live cached graph.
    """
    global _MEDIAPIPE_COOLDOWN_UNTIL, _MEDIAPIPE_FAILURES, _MEDIAPIPE_LAST_ERROR
    with _MEDIAPIPE_LOCK:
        _MEDIAPIPE_COOLDOWN_UNTIL = 0.0
        _MEDIAPIPE_FAILURES = 0
        _MEDIAPIPE_LAST_ERROR = None
        _close_mediapipe_sessions_locked()


def _ensure_face_detector_locked():
    """Return a live FaceDetection graph, creating it once per worker."""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
    return _FACE_DETECTOR


def _ensure_face_mesh_locked():
    """Return a live FaceMesh graph, creating it once per worker."""
    global _FACE_MESH
    if _FACE_MESH is None:
        _FACE_MESH = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _FACE_MESH


def _int_env(name, default, *, minimum, maximum):
    """Read a bounded integer environment variable, falling back on bad input."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "Invalid %s=%r; using default %s.", name, raw, default
        )
        return default
    if value < minimum or value > maximum:
        logger.warning(
            "%s=%s out of range [%s, %s]; using default %s.",
            name, value, minimum, maximum, default,
        )
        return default
    return value


# Upper bound on decoded pixels. A 12 MB upload can still expand into a
# multi-gigapixel bitmap, so the compressed size limit alone is not enough
# protection against decompression bombs.
# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
MAX_IMAGE_PIXELS = _int_env(
    "MAX_IMAGE_PIXELS", 40_000_000, minimum=1_000_000, maximum=500_000_000
)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Only formats the portal actually accepts are decoded.
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


class ImageValidationError(ValueError):
    """Raised when an upload cannot be safely decoded as a portrait photo."""


# ---------- Helpers ----------

def load_validated_image(image_bytes):
    """Decode upload bytes into an EXIF-corrected RGB ``PIL.Image``.

    Raises ``ImageValidationError`` with a caller-safe message when the payload
    is empty, is not an allowed image format, is corrupt, or would decode into
    an unreasonably large bitmap.
    """
    if not image_bytes:
        raise ImageValidationError("Empty image payload.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("Image is too large to process safely.") from exc
    except Exception as exc:
        raise ImageValidationError("Unsupported or corrupt image file.") from exc

    image_format = (pil_img.format or "").upper()
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise ImageValidationError(
            "Unsupported image format. Allowed formats: JPEG, PNG, WEBP."
        )

    width, height = pil_img.size
    if width <= 0 or height <= 0:
        raise ImageValidationError("Image has invalid dimensions.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageValidationError(
            f"Image is too large ({width}x{height}). "
            f"Maximum supported size is {MAX_IMAGE_PIXELS} pixels."
        )

    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        # Orientation metadata is best-effort; a broken EXIF block must not
        # prevent an otherwise valid photo from being verified.
        logger.debug("EXIF transpose failed; using raw orientation.")

    try:
        return pil_img.convert("RGB")
    except Exception as exc:
        raise ImageValidationError("Image could not be decoded.") from exc


def _load_image(image_bytes):
    """Load image bytes into a validated OpenCV BGR array."""
    pil_img = load_validated_image(image_bytes)
    arr = np.array(pil_img)
    if arr.size == 0:
        raise ImageValidationError("Image could not be decoded.")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _nms_faces(faces, h, w, iou_threshold=0.35, min_area_ratio=0.03):
    """
    Non-Maximum Suppression & overlap filtering to eliminate duplicate face bounding boxes.
    Merges boxes that overlap significantly (IoU > iou_threshold or intersection over min > 0.45),
    and filters out tiny false positive boxes (< 3% of total image area unless it is the only detection).
    """
    if not faces:
        return []

    img_area = float(w * h)
    valid_faces = []
    for f in faces:
        area = f["w"] * f["h"]
        if len(faces) > 1 and (area / img_area) < min_area_ratio:
            continue
        valid_faces.append(f)

    if not valid_faces:
        valid_faces = faces

    valid_faces = sorted(valid_faces, key=lambda f: (f["w"] * f["h"]) * f.get("score", 1.0), reverse=True)

    keep = []
    for f in valid_faces:
        f_area = float(f["w"] * f["h"])
        is_duplicate = False

        for k in keep:
            k_area = float(k["w"] * k["h"])
            ix1 = max(f["x"], k["x"])
            iy1 = max(f["y"], k["y"])
            ix2 = min(f["x"] + f["w"], k["x"] + k["w"])
            iy2 = min(f["y"] + f["h"], k["y"] + k["h"])

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = float(iw * ih)

            union = f_area + k_area - inter
            iou = inter / union if union > 0 else 0.0
            min_area = min(f_area, k_area)
            iom = inter / min_area if min_area > 0 else 0.0

            if iou > iou_threshold or iom > 0.45:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(f)

    return keep


def _detect_faces(bgr, meta_out: dict = None) -> list[dict]:
    """Return list of face detections with bounding boxes (relative + absolute).
    Uses mediapipe when available, otherwise falls back to OpenCV's Haar cascade.
    Applies NMS to filter out duplicate overlapping boxes and tiny false positives."""
    h, w = bgr.shape[:2]

    faces = []
    mediapipe_failed = False
    if _mediapipe_usable():
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with _MEDIAPIPE_LOCK:
                fd = _ensure_face_detector_locked()
                results = fd.process(rgb)
                if results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        x = max(0, int(bbox.xmin * w))
                        y = max(0, int(bbox.ymin * h))
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        faces.append({
                            "x": x, "y": y, "w": bw, "h": bh,
                            "score": float(det.score[0]) if det.score else 0.0,
                            "keypoints": det.location_data.relative_keypoints
                        })
            _note_mediapipe_success()
            if meta_out is not None:
                meta_out["face_backend"] = "MediaPipe"
        except Exception as exc:
            # Fall back to Haar cascades for THIS image only, and open a short
            # cooldown. Drop the cached graph so the next retry constructs a
            # fresh one rather than reusing a broken session.
            logger.warning(
                "MediaPipe failed (%s); using Haar Cascade for this image and "
                "retrying MediaPipe in %.0fs.",
                type(exc).__name__, _MEDIAPIPE_COOLDOWN_SECONDS,
            )
            mediapipe_failed = True
            with _MEDIAPIPE_LOCK:
                _close_mediapipe_sessions_locked()
            _note_mediapipe_failure(exc)

    if mediapipe_failed or not _mediapipe_usable():
        # Haar cascade fallback
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = _HAAR_FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        for (x, y, fw, fh) in detections:
            faces.append({"x": int(x), "y": int(y), "w": int(fw), "h": int(fh), "score": 1.0, "keypoints": None})
        if meta_out is not None:
            meta_out["face_backend"] = "haar cascade"   
    return _nms_faces(faces, h, w)


def _face_mesh_landmarks(bgr):
    """Fine-grained facial landmarks via mediapipe FaceMesh.

    Returns None when landmarks genuinely cannot be produced. Callers treat
    that as "evidence unavailable" and route to manual review, so a MediaPipe
    runtime error must not be allowed to persist beyond its cooldown.

    Head pose and eyes-open both need the same landmarks. The result is cached
    on the identity of ``bgr`` for the rest of this request so FaceMesh runs
    once per photograph, not once per check.
    """
    global _LANDMARK_CACHE_ID, _LANDMARK_CACHE_VAL
    if not _mediapipe_usable():
        return None
    cache_key = id(bgr)
    with _MEDIAPIPE_LOCK:
        if cache_key == _LANDMARK_CACHE_ID:
            return _LANDMARK_CACHE_VAL
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        with _MEDIAPIPE_LOCK:
            if cache_key == _LANDMARK_CACHE_ID:
                return _LANDMARK_CACHE_VAL
            fm = _ensure_face_mesh_locked()
            results = fm.process(rgb)
            if not results.multi_face_landmarks:
                pts = None
            else:
                lm = results.multi_face_landmarks[0]
                pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
            _LANDMARK_CACHE_ID = cache_key
            _LANDMARK_CACHE_VAL = pts
        _note_mediapipe_success()
        return pts
    except Exception as exc:
        logger.warning(
            "FaceMesh failed (%s); landmark checks will require manual review "
            "and MediaPipe will be retried in %.0fs.",
            type(exc).__name__, _MEDIAPIPE_COOLDOWN_SECONDS,
        )
        with _MEDIAPIPE_LOCK:
            _close_mediapipe_sessions_locked()
        _note_mediapipe_failure(exc)
        return None


def _extract_face_skin_profile(bgr, pts=None, face=None):
    """
    Extracts the individual subject's facial skin color profile in LAB & BGR.
    Samples safe facial regions: inner cheeks and forehead away from eyes, hair, and lips.
    """
    h, w = bgr.shape[:2]
    sampled_pixels = []

    if pts is not None:
        safe_indices = [10, 151, 9, 8, 50, 117, 123, 205, 280, 346, 352, 425]
        for idx in safe_indices:
            if idx < len(pts):
                px, py = pts[idx]
                if 0 <= px < w and 0 <= py < h:
                    y1, y2 = max(0, py - 1), min(h, py + 2)
                    x1, x2 = max(0, px - 1), min(w, px + 2)
                    patch = bgr[y1:y2, x1:x2]
                    if patch.size > 0:
                        sampled_pixels.extend(patch.reshape(-1, 3))
    elif face is not None:
        fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]
        lc = bgr[max(0, fy + int(fh * 0.45)):min(h, fy + int(fh * 0.65)), max(0, fx + int(fw * 0.15)):min(w, fx + int(fw * 0.35))]
        rc = bgr[max(0, fy + int(fh * 0.45)):min(h, fy + int(fh * 0.65)), max(0, fx + int(fw * 0.65)):min(w, fx + int(fw * 0.85))]
        fh_roi = bgr[max(0, fy + int(fh * 0.15)):min(h, fy + int(fh * 0.30)), max(0, fx + int(fw * 0.35)):min(w, fx + int(fw * 0.65))]
        for roi in [lc, rc, fh_roi]:
            if roi.size > 0:
                sampled_pixels.extend(roi.reshape(-1, 3))

    if not sampled_pixels:
        skin_bgr_mean = np.array([110.0, 130.0, 170.0], dtype=np.float32)
    else:
        skin_bgr_mean = np.median(np.array(sampled_pixels, dtype=np.float32), axis=0)

    single_pixel = np.uint8([[skin_bgr_mean]])
    skin_lab_mean = cv2.cvtColor(single_pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    return {
        "bgr_mean": skin_bgr_mean,
        "lab_mean": skin_lab_mean,
    }


def _compute_skin_distance(bgr_patch, skin_lab_mean):
    """Computes perceptual color distance from subject's calibrated skin tone."""
    if bgr_patch.size == 0:
        return np.array([])
    lab = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    dL = lab[:, :, 0] - skin_lab_mean[0]
    da = lab[:, :, 1] - skin_lab_mean[1]
    db = lab[:, :, 2] - skin_lab_mean[2]
    # Perceptual color distance downweighting luminance for illumination tolerance
    return np.sqrt(0.25 * (dL ** 2) + (da ** 2) + (db ** 2))


# ---------- Individual Checks ----------

def check_face_count(bgr, faces, params):
    n = len(faces)
    if n == 0:
        return {"passed": False, "message": "No face detected in the photo. Please upload a clear photo of your face.", "meta": {"count": n}}
    if n > 1:
        return {"passed": False, "message": f"Multiple faces detected ({n}). Only one person should be in the photo.", "meta": {"count": n}}
    return {"passed": True, "message": "Single face detected.", "meta": {"count": n}}


def check_face_size_centering(bgr, faces, params):
    if not faces:
        return {"passed": False, "message": "Face not detected — cannot verify framing.", "meta": {}}
    h, w = bgr.shape[:2]
    f = faces[0]
    face_area_ratio = (f["w"] * f["h"]) / float(w * h)
    cx = f["x"] + f["w"] / 2.0
    cy = f["y"] + f["h"] / 2.0
    center_dx = abs(cx - w / 2.0) / w
    center_dy = abs(cy - h / 2.0) / h

    min_ratio = params.get("min_face_ratio", 0.10)
    max_ratio = params.get("max_face_ratio", 0.60)
    max_offset = params.get("max_center_offset", 0.18)

    issues = []
    if face_area_ratio < min_ratio:
        issues.append("Face is too small / too far from camera.")
    if face_area_ratio > max_ratio:
        issues.append("Face is too large / too close to camera.")
    if center_dx > max_offset or center_dy > max_offset:
        issues.append("Face is not centered in the frame.")

    if issues:
        return {"passed": False, "message": " ".join(issues), "meta": {"face_area_ratio": face_area_ratio}}
    return {"passed": True, "message": "Face size and centering look good.", "meta": {"face_area_ratio": face_area_ratio}}


def check_glasses(bgr, faces, params):
    """
    High-precision Eyeglasses Detector.
    Uses multi-signal anatomical corroboration:
    1. Inter-ocular Nose Bridge Bar analysis (horizontal gradient Gy, Canny edge density, and skin color distance).
    2. Infraorbital Lower Cheek Rim analysis (gradient energy on the upper cheekbone below the orbital rim).
    3. Supraorbital & lateral frame hinge corroboration.
    Eliminates false positives on bare eyes and natural facial features while reliably catching frames.
    """
    if not faces:
        return {"passed": False, "message": "Face not detected — cannot check for glasses.", "meta": {}}

    h, w = bgr.shape[:2]
    f = faces[0]
    fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]

    pts = _face_mesh_landmarks(bgr)
    skin = _extract_face_skin_profile(bgr, pts, f)

    bridge_edge_y = 0.0
    bridge_canny = 0.0
    bridge_skin_diff = 0.0
    infra_edge = 0.0
    infra_min = 0.0
    lm_analyzed = False

    if pts is not None:
        try:
            lm_analyzed = True
            l_in, r_in = pts[133], pts[362]
            l_bot, r_bot = pts[145], pts[374]
            eye_span = max(10, abs(r_in[0] - l_in[0]))

            # 1. Nose Bridge ROI between inner eye corners
            bx1 = max(0, l_in[0] + int(eye_span * 0.15))
            bx2 = min(w, r_in[0] - int(eye_span * 0.15))
            by1 = max(0, min(l_in[1], r_in[1]) - int(eye_span * 0.20))
            by2 = min(h, max(l_in[1], r_in[1]) + int(eye_span * 0.20))
            b_crop = bgr[by1:by2, bx1:bx2]

            if b_crop.size > 0:
                b_gray = cv2.cvtColor(b_crop, cv2.COLOR_BGR2GRAY)
                bridge_edge_y = float(np.mean(np.abs(cv2.Sobel(b_gray, cv2.CV_64F, 0, 1, ksize=3))))
                bridge_canny = float(np.mean(cv2.Canny(b_crop, 40, 110) > 0))
                b_dists = _compute_skin_distance(b_crop, skin["lab_mean"])
                bridge_skin_diff = float(np.mean(b_dists))

            # 2. Infraorbital cheek patches centered on the cheekbone below the lower eye crease
            ly1 = l_bot[1] + int(eye_span * 0.30)
            ly2 = min(h, l_bot[1] + int(eye_span * 0.58))
            lx1 = max(0, pts[159][0] - int(eye_span * 0.15))
            lx2 = min(w, pts[159][0] + int(eye_span * 0.15))
            l_patch = bgr[ly1:ly2, lx1:lx2]

            ry1 = r_bot[1] + int(eye_span * 0.30)
            ry2 = min(h, r_bot[1] + int(eye_span * 0.58))
            rx1 = max(0, pts[386][0] - int(eye_span * 0.15))
            rx2 = min(w, pts[386][0] + int(eye_span * 0.15))
            r_patch = bgr[ry1:ry2, rx1:rx2]

            l_edge = float(np.mean(np.abs(cv2.Sobel(cv2.cvtColor(l_patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)))) if l_patch.size else 0.0
            r_edge = float(np.mean(np.abs(cv2.Sobel(cv2.cvtColor(r_patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 0, 1, ksize=3)))) if r_patch.size else 0.0
            infra_edge = (l_edge + r_edge) / 2.0
            infra_min = min(l_edge, r_edge)
        except Exception:
            lm_analyzed = False

    if not lm_analyzed:
        # Fallback based on face bounding box geometry
        bx1 = max(0, fx + int(fw * 0.40))
        bx2 = min(w, fx + int(fw * 0.60))
        by1 = max(0, fy + int(fh * 0.28))
        by2 = min(h, fy + int(fh * 0.40))
        b_crop = bgr[by1:by2, bx1:bx2]
        if b_crop.size > 0:
            b_gray = cv2.cvtColor(b_crop, cv2.COLOR_BGR2GRAY)
            bridge_edge_y = float(np.mean(np.abs(cv2.Sobel(b_gray, cv2.CV_64F, 0, 1, ksize=3))))
            bridge_canny = float(np.mean(cv2.Canny(b_crop, 40, 110) > 0))
            b_dists = _compute_skin_distance(b_crop, skin["lab_mean"])
            bridge_skin_diff = float(np.mean(b_dists))

    # Tunable thresholds
    th_bridge_edge = float(params.get("glasses_bridge_edge", 23.0))
    th_bridge_diff = float(params.get("glasses_bridge_diff", 18.0))
    th_cheek_edge = float(params.get("glasses_cheek_edge", 24.0))

    # Multi-signal decision matrix:
    # 1. Distinct horizontal nose bridge bar with non-skin color or edge density
    sig_bridge = (bridge_edge_y >= th_bridge_edge) and (bridge_skin_diff >= th_bridge_diff or bridge_canny >= 0.05)
    # 2. Strong frame edges across both cheekbones below lower eyelids
    sig_cheek = (infra_min >= 20.0) and (infra_edge >= th_cheek_edge)
    # 3. Corroborating bridge structure + lower rim cheek edges
    sig_combo = (bridge_edge_y >= 18.0 or bridge_skin_diff >= 20.0) and (infra_edge >= 18.0)

    glasses_detected = sig_bridge or sig_cheek or sig_combo

    meta = {
        "bridge_edge_y": round(bridge_edge_y, 2),
        "bridge_canny": round(bridge_canny, 4),
        "bridge_skin_diff": round(bridge_skin_diff, 2),
        "infraorbital_edge": round(infra_edge, 2),
        "lm_analyzed": lm_analyzed
    }

    if glasses_detected:
        return {"passed": False, "message": "Eyeglasses detected. Please remove glasses for your passport photo.", "meta": meta}
    return {"passed": True, "message": "No eyeglasses detected.", "meta": meta}


# Default white-background threshold fallbacks
DEFAULT_WHITE_BACKGROUND_PARAMS = {
    # Tier 1: pure studio white
    "bg_min_value": 235.0,
    "bg_max_saturation": 18.0,
    "bg_max_delta_e": 10.0,
    # Tier 2: near-white
    "bg_near_white_enabled": 0.0,
    "bg_near_white_min_l_star": 93.0,
    "bg_near_white_max_chroma": 10.0,
    "bg_near_white_max_b_star": 9.0,
    # Coverage & uniformity thresholds
    "bg_min_white_coverage": 30.0,
    "bg_max_nonwhite_component_coverage": 30.0,
    "bg_max_luminance_range": 100.0,
    "bg_reject_dark_value": 210.0,
    "bg_max_dark_coverage": 5.0,
    "bg_reject_colored_saturation": 30.0,
    "bg_max_colored_coverage": 5.0,
    "bg_border_fraction": 0.12,
}


# Strictness level presets for background validation
BACKGROUND_STRICTNESS_LEVELS = {
    "strict": {
        "bg_min_value": 235.0,
        "bg_max_saturation": 18.0,
        "bg_max_delta_e": 10.0,
        "bg_near_white_enabled": 0.0,
        "bg_near_white_min_l_star": 93.0,
        "bg_near_white_max_chroma": 10.0,
        "bg_near_white_max_b_star": 9.0,
        "bg_min_white_coverage": 30.0,
        "bg_max_nonwhite_component_coverage": 30.0,
        "bg_max_luminance_range": 100.0,
        "bg_reject_dark_value": 210.0,
        "bg_max_dark_coverage": 5.0,
        "bg_reject_colored_saturation": 30.0,
        "bg_max_colored_coverage": 5.0,
        "bg_border_fraction": 0.12,
    },
    "standard": {
        # Balanced lighting tolerance for indoor white walls
        "bg_min_value": 150.0,
        "bg_max_saturation": 25.0,
        "bg_max_delta_e": 38.0,
        "bg_near_white_enabled": 1.0,
        "bg_near_white_min_l_star": 60.0,
        "bg_near_white_max_chroma": 13.0,
        "bg_near_white_max_b_star": 13.0,
        "bg_min_white_coverage": 60.0,
        "bg_max_nonwhite_component_coverage": 30.0,
        "bg_max_luminance_range": 130.0,
        "bg_reject_dark_value": 120.0,
        "bg_max_dark_coverage": 8.0,
        "bg_reject_colored_saturation": 36.0,
        "bg_max_colored_coverage": 5.0,
        "bg_border_fraction": 0.12,
    },
    "relaxed": {
        # Relaxed thresholds for home-taken photos
        "bg_min_value": 135.0,
        "bg_max_saturation": 28.0,
        "bg_max_delta_e": 44.0,
        "bg_near_white_enabled": 1.0,
        "bg_near_white_min_l_star": 52.0,
        "bg_near_white_max_chroma": 13.0,
        "bg_near_white_max_b_star": 13.0,
        "bg_min_white_coverage": 50.0,
        "bg_max_nonwhite_component_coverage": 40.0,
        "bg_max_luminance_range": 160.0,
        "bg_reject_dark_value": 105.0,
        "bg_max_dark_coverage": 12.0,
        "bg_reject_colored_saturation": 38.0,
        "bg_max_colored_coverage": 5.0,
        "bg_border_fraction": 0.12,
    },
    "accept_all": {
        # Only rejects dark or vividly colored backgrounds
        "bg_min_value": 110.0,
        "bg_max_saturation": 80.0,
        "bg_max_delta_e": 45.0,
        "bg_near_white_enabled": 1.0,
        "bg_near_white_min_l_star": 80.0,
        "bg_near_white_max_chroma": 30.0,
        "bg_near_white_max_b_star": 28.0,
        "bg_min_white_coverage": 5.0,
        "bg_max_nonwhite_component_coverage": 80.0,
        "bg_max_luminance_range": 200.0,
        "bg_reject_dark_value": 100.0,
        "bg_max_dark_coverage": 20.0,
        "bg_reject_colored_saturation": 80.0,
        "bg_max_colored_coverage": 20.0,
        "bg_border_fraction": 0.12,
    },
}


# Canonical values for the near-white acceptance switch
NEAR_WHITE_ACCEPTANCE_VALUES = ("auto", "1", "0")


def _resolve_background_params(params):
    """Resolve and merge background strictness preset parameters."""
    level = str(params.get("background_strictness", "standard")).strip().lower()
    if level not in BACKGROUND_STRICTNESS_LEVELS:
        level = "standard"
    preset = BACKGROUND_STRICTNESS_LEVELS[level]

    merged = dict(preset)
    merged["background_strictness"] = level

    raw_switch = params.get(
        "background_near_white_acceptance",
        params.get("bg_near_white_enabled", "auto"),
    )
    if isinstance(raw_switch, bool):
        switch = "1" if raw_switch else "0"
    elif isinstance(raw_switch, (int, float)):
        # Numeric 0/0.0 -> off, any non-zero number -> on (never "auto").
        switch = "1" if float(raw_switch) != 0.0 else "0"
    else:
        switch = str(raw_switch).strip().lower() or "auto"
        if switch not in NEAR_WHITE_ACCEPTANCE_VALUES:
            switch = "auto"
    merged["bg_near_white_enabled"] = 1.0 if switch == "1" else 0.0 if switch == "0" \
        else float(preset["bg_near_white_enabled"])
    merged["background_near_white_acceptance"] = switch

    return merged


def _white_bg_param(params, key):
    """Read a preset-supplied background parameter defensively.

    Values come exclusively from ``_resolve_background_params`` (level presets
    are trusted code constants), but this keeps malformed runtime input from
    ever producing NaN/inf thresholds.
    """
    default = DEFAULT_WHITE_BACKGROUND_PARAMS[key]
    try:
        value = float(params.get(key, default))
        if not np.isfinite(value):
            value = default
    except (TypeError, ValueError):
        value = default
    return value


def _background_border_mask(shape, faces, border_fraction):
    """Return outer background regions while excluding the detected subject area.

    The detector samples visible background from the upper border and side bands
    while properly enveloping the head (including hair volume) and upper torso.
    A scale-adaptive morphological margin is applied to insulate the sample against
    edge-blur and downsampling interpolation artifacts on low-resolution devices.
    Extreme corners are inset to eliminate phone screenshot anti-aliased bezels.
    """
    h, w = shape
    border_x = max(1, int(round(w * border_fraction)))
    border_y = max(1, int(round(h * border_fraction)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:border_y, :] = 1

    valid_faces = []
    for face in faces or []:
        try:
            fx, fy = int(face["x"]), int(face["y"])
            fw, fh = int(face["w"]), int(face["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if fw > 0 and fh > 0:
            valid_faces.append((fx, fy, fw, fh))

    if not valid_faces:
        # Sample the top border and upper side strips when no face is found
        upper_side_limit = max(1, int(h * 0.40))
        mask[:upper_side_limit, :border_x] = 1
        mask[:upper_side_limit, w - border_x:] = 1
    else:
        # Sample the side borders along the frame height
        mask[:, :border_x] = 1
        mask[:, w - border_x:] = 1

    # Inset corners to avoid rounded corners/bezels
    corner_inset = max(3, int(min(w, h) * 0.015))
    mask[:corner_inset, :corner_inset] = 0
    mask[:corner_inset, w - corner_inset:] = 0
    mask[h - corner_inset:, :corner_inset] = 0
    mask[h - corner_inset:, w - corner_inset:] = 0

    if valid_faces:
        subject_mask = np.zeros((h, w), dtype=np.uint8)
        for fx, fy, fw, fh in valid_faces:
            center_x = int(fx + fw * 0.5)
            center_y = int(fy + fh * 0.40)
            cv2.ellipse(
                subject_mask,
                (center_x, center_y),
                (max(1, int(fw * 0.95)), max(1, int(fh * 1.15))),
                0,
                0,
                360,
                1,
                thickness=-1,
            )
            torso_top = min(h - 1, max(0, int(fy + fh * 0.60)))
            torso_bottom = h - 1
            torso = np.array([
                [max(0, int(center_x - fw * 0.90)), torso_top],
                [min(w - 1, int(center_x + fw * 0.90)), torso_top],
                [min(w - 1, int(center_x + fw * 2.50)), torso_bottom],
                [max(0, int(center_x - fw * 2.50)), torso_bottom],
            ], dtype=np.int32)
            cv2.fillConvexPoly(subject_mask, torso, 1)

        # Dilate subject mask to prevent border edge bleeding
        margin = max(2, int(round(min(w, h) * 0.015)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        dilated_subject = cv2.dilate(subject_mask, kernel)
        mask[dilated_subject > 0] = 0

    return mask


def _largest_component_coverage(binary_mask, sample_mask, sample_count):
    """Measure the largest contiguous contaminated background region."""
    if sample_count == 0 or not np.any(binary_mask):
        return 0.0
    component_input = np.where(sample_mask, binary_mask, 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(component_input, connectivity=8)
    if count <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest / float(sample_count)


def _measure_sharpness(bgr, faces):
    """Compute face-ROI Laplacian variance."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if faces:
        f = faces[0]
        h, w = gray.shape[:2]
        pad_x = int(f["w"] * 0.25)
        pad_y = int(f["h"] * 0.25)
        x1 = max(0, f["x"] - pad_x)
        y1 = max(0, f["y"] - pad_y)
        x2 = min(w, f["x"] + f["w"] + pad_x)
        y2 = min(h, f["y"] + f["h"] + pad_y)
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            return float(cv2.Laplacian(roi, cv2.CV_64F).var())
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_white_background(bgr, faces, params):
    """Verifies that the background is white and uniform against preset thresholds."""
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim != 3:
        return {"passed": False, "message": "Invalid image data.", "meta": {}}
    h, w = bgr.shape[:2]
    if h < 2 or w < 2:
        return {"passed": False, "message": "Invalid image dimensions.", "meta": {}}

    params = _resolve_background_params(params)

    min_value = _white_bg_param(params, "bg_min_value")
    max_saturation = _white_bg_param(params, "bg_max_saturation")
    max_delta_e = _white_bg_param(params, "bg_max_delta_e")
    near_white_enabled = _white_bg_param(params, "bg_near_white_enabled") >= 0.5
    near_white_min_l_star = _white_bg_param(params, "bg_near_white_min_l_star")
    near_white_max_chroma = _white_bg_param(params, "bg_near_white_max_chroma")
    near_white_max_b_star = _white_bg_param(params, "bg_near_white_max_b_star")
    min_white_coverage = _white_bg_param(params, "bg_min_white_coverage")
    max_component_coverage = _white_bg_param(params, "bg_max_nonwhite_component_coverage")
    max_luminance_range = _white_bg_param(params, "bg_max_luminance_range")
    reject_dark_value = _white_bg_param(params, "bg_reject_dark_value")
    max_dark_coverage = _white_bg_param(params, "bg_max_dark_coverage")
    reject_colored_saturation = _white_bg_param(params, "bg_reject_colored_saturation")
    max_colored_coverage = _white_bg_param(params, "bg_max_colored_coverage")
    border_fraction = min(0.30, max(0.03, _white_bg_param(params, "bg_border_fraction")))

    sample_mask = _background_border_mask((h, w), faces, border_fraction).astype(bool)

    if bgr.ndim == 3 and bgr.shape[2] == 4:
        alpha_chan = bgr[:, :, 3]
        bgr = bgr[:, :, :3]
        sample_mask = sample_mask & (alpha_chan < 25)

    sample_count = int(np.count_nonzero(sample_mask))
    if sample_count < 32:
        return {
            "passed": False,
            "message": "White background not accepted, please try again.",
            "meta": {"sampled_pixels": sample_count, "minimum_sampled_pixels": 32, "issues": ["Could not find enough visible background to verify a white background."]},
        }

    # Median filter to reduce sensor noise
    ksize = 3 if min(w, h) < 600 else 5
    denoised_bgr = cv2.medianBlur(bgr, ksize)

    hsv = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)
    bgr_float = denoised_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr_float, cv2.COLOR_BGR2LAB)

    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    delta_e = np.sqrt((100.0 - L) ** 2 + a ** 2 + b ** 2)

    # Tier 1: Pure white check
    tier_pure = (value >= min_value) & (saturation <= max_saturation) & (delta_e <= max_delta_e)

    # Tier 2: Near-white check for realistic lighting
    if near_white_enabled:
        chroma = np.sqrt(a.astype(np.float32) ** 2 + b.astype(np.float32) ** 2)
        tier_near_white = (
            (L >= near_white_min_l_star)
            & (chroma <= near_white_max_chroma)
            & (np.abs(b) <= near_white_max_b_star)
        )
    else:
        tier_near_white = np.zeros(value.shape, dtype=bool)

    white_pixels = (tier_pure | tier_near_white) & sample_mask

    # Pure studio white probe for metadata reporting
    pure_white_pixels = (value >= 235) & (saturation <= 20) & (delta_e <= 12) & sample_mask
    pure_white_count = int(np.count_nonzero(pure_white_pixels))

    nonwhite_mask = (~white_pixels) & sample_mask

    # Identify dark and colored pixels in sampled background
    effective_dark_cutoff = min(140.0, reject_dark_value)
    dark_mask = (value < effective_dark_cutoff) & sample_mask
    colored_mask = (
        (saturation > reject_colored_saturation)
        & (value >= effective_dark_cutoff)
        & sample_mask
    )

    sampled_value = value[sample_mask]
    sampled_saturation = saturation[sample_mask]
    sampled_delta_e = delta_e[sample_mask]
    sampled_L = L[sample_mask]

    nonwhite_count = int(np.count_nonzero(nonwhite_mask))
    nonwhite_coverage = nonwhite_count / float(sample_count) * 100.0
    white_coverage = 100.0 - nonwhite_coverage
    near_white_count = int(np.count_nonzero(tier_near_white & sample_mask & ~tier_pure))
    near_white_coverage = near_white_count / float(sample_count) * 100.0

    component_coverage = _largest_component_coverage(nonwhite_mask, sample_mask, sample_count) * 100.0
    dark_coverage = float(np.count_nonzero(dark_mask)) / float(sample_count) * 100.0
    colored_coverage = float(np.count_nonzero(colored_mask)) / float(sample_count) * 100.0
    luminance_range = float(np.percentile(sampled_L, 90) - np.percentile(sampled_L, 10))

    issues = []
    if white_coverage < min_white_coverage:
        issues.append(
            f"White background coverage is {white_coverage:.2f}% (minimum {min_white_coverage:.2f}%)."
        )
    if component_coverage > max_component_coverage:
        issues.append(
            f"A contiguous non-white background region covers {component_coverage:.2f}% (allowed {max_component_coverage:.2f}%)."
        )
    if dark_coverage > max_dark_coverage:
        issues.append(
            f"Dark or black background pixels cover {dark_coverage:.2f}% (allowed {max_dark_coverage:.2f}%)."
        )
    if colored_coverage > max_colored_coverage:
        issues.append(
            f"Coloured background pixels cover {colored_coverage:.2f}% (allowed {max_colored_coverage:.2f}%)."
        )
    if luminance_range > max_luminance_range:
        issues.append(
            f"Background luminance varies by {luminance_range:.2f} L* (allowed {max_luminance_range:.2f}); shadows or a gradient were detected."
        )

    score = max(0.0, 100.0 - nonwhite_coverage - min(100.0, float(np.mean(sampled_delta_e))))
    thresholds = {
        "min_value": min_value,
        "max_saturation": max_saturation,
        "max_delta_e": max_delta_e,
        "near_white_enabled": bool(near_white_enabled),
        "near_white_min_l_star": near_white_min_l_star,
        "near_white_max_chroma": near_white_max_chroma,
        "near_white_max_b_star": near_white_max_b_star,
        "min_white_coverage_percent": min_white_coverage,
        "max_nonwhite_component_coverage_percent": max_component_coverage,
        "max_luminance_range_l_star": max_luminance_range,
        "reject_dark_value": reject_dark_value,
        "max_dark_coverage_percent": max_dark_coverage,
        "reject_colored_saturation": reject_colored_saturation,
        "max_colored_coverage_percent": max_colored_coverage,
        "border_fraction": border_fraction,
        "luminance_range_percentiles": [10, 90],
        "blur_tolerance_applied": False,
        "background_strictness": params.get("background_strictness", "standard"),
        "background_near_white_acceptance": params.get(
            "background_near_white_acceptance", "auto"
        ),
    }
    meta = {
        "sampled_pixels": sample_count,
        "white_coverage_percent": round(white_coverage, 3),
        "pure_white_coverage_percent": round(pure_white_count / float(sample_count) * 100.0, 3),
        "near_white_coverage_percent": round(near_white_coverage, 3),
        "nonwhite_coverage_percent": round(nonwhite_coverage, 3),
        "largest_nonwhite_component_percent": round(component_coverage, 3),
        "dark_coverage_percent": round(dark_coverage, 3),
        "colored_coverage_percent": round(colored_coverage, 3),
        "mean_value": round(float(np.mean(sampled_value)), 3),
        "mean_l_star": round(float(np.mean(sampled_L)), 3),
        "p10_l_star": round(float(np.percentile(sampled_L, 10)), 3),
        "p90_l_star": round(float(np.percentile(sampled_L, 90)), 3),
        "max_saturation_detected": round(float(np.max(sampled_saturation)), 3),
        "mean_delta_e_to_white": round(float(np.mean(sampled_delta_e)), 3),
        "max_delta_e_to_white": round(float(np.max(sampled_delta_e)), 3),
        "luminance_range_l_star": round(luminance_range, 3),
        "quality_score": round(score, 3),
        "thresholds": thresholds,
    }
    if not issues:
        return {"passed": True, "message": "White bg accepted.", "meta": meta}
    meta["issues"] = issues
    return {"passed": False, "message": "White background not accepted, please try again.", "meta": meta}


# Cached estimator instance — created once per worker.
_upper_body_estimator = UpperBodyVisibilityEstimator()

# Geometry ranges for tie validation
_TEST_TIE_GEOMETRY = {
    "min_width_face_ratio": 0.04,
    "max_width_face_ratio": 0.85,
    "min_height_face_ratio": 0.06,
    "max_height_face_ratio": 1.60,
    "min_top_offset_face_ratio": -0.45,
    "max_top_offset_face_ratio": 1.35,
    "max_center_offset_face_ratio": 0.40,
}


def _tie_positive_policy(detector, params):
    """Return positive decision thresholds and geometry policy for tie detector."""
    policy = getattr(detector, "policy", None)
    if isinstance(policy, TieModelPolicy):
        return policy.positive_threshold, policy.geometry, policy.model_version
    coco_threshold = getattr(detector, "positive_threshold", None)
    coco_geometry = getattr(detector, "geometry", None)
    if isinstance(coco_threshold, (int, float)) and isinstance(coco_geometry, dict):
        return float(coco_threshold), coco_geometry, detector.version
    return (
        float(params.get("tie_require_threshold", params.get("tie_reject_threshold", 0.65))),
        _TEST_TIE_GEOMETRY,
        os.environ.get("TIE_MODEL_VERSION", "tie-detector-dev"),
    )


def _tie_manual_review(message, *, status, visible, model_version, **meta):
    return {
        "passed": False,
        "message": message,
        "meta": {
            "decision": "manual_review",
            "tie_status": status,
            "tie_detected": None,
            "upper_body_visible": visible,
            "model_version": model_version,
            **meta,
        },
    }


def _tie_absent_result(model_version, *, reason, confidence=0.0, bbox=None):
    """Return a required-tie failure after a validated negative decision."""
    meta = {
        "decision": "reject",
        "tie_status": "tie_absent",
        "tie_detected": False,
        "confidence": round(confidence, 4),
        "upper_body_visible": True,
        "reason": reason,
        "model_version": model_version,
    }
    if bbox is not None:
        meta["bbox"] = bbox
    return {
        "passed": False,
        "message": "Traditional necktie not detected in the visible neck/chest region.",
        "meta": meta,
    }


def _illumination_robust_tie_cues(chest_crop):
    """Measure dark-on-dark structure without rewriting the stored photograph.

    CLAHE is applied to a copy of the L channel only. The original BGR crop is
    never mutated, so enhancement cannot invent a garment that is not there —
    it only makes existing local edges measurable when mean RGB contrast is low.
    """
    empty = {
        "clahe_vert_edge_score": 0.0,
        "local_l_contrast": 0.0,
        "ridge_width_frac": 0.0,
        "horizontal_bimodality": 0.0,
    }
    if chest_crop is None or chest_crop.size == 0:
        return empty
    ch, cw = chest_crop.shape[:2]
    if ch < 8 or cw < 16:
        return empty

    lab = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    scharr_x = np.abs(cv2.Scharr(l_eq, cv2.CV_64F, 1, 0))
    v_left = scharr_x[:, int(cw * 0.28):int(cw * 0.42)]
    v_right = scharr_x[:, int(cw * 0.58):int(cw * 0.72)]
    clahe_vert = 0.0
    if v_left.size and v_right.size:
        clahe_vert = (float(np.mean(v_left)) + float(np.mean(v_right))) / 2.0 / 18.0

    c_l = l_eq[:, int(cw * 0.36):int(cw * 0.64)]
    local_l = float(np.std(c_l)) if c_l.size else 0.0

    widths = []
    for row in l_eq:
        fl = float(np.mean(row[: max(1, int(cw * 0.22))]))
        fr = float(np.mean(row[int(cw * 0.78):]))
        flank = (fl + fr) / 2.0
        diff = np.abs(row.astype(np.float32) - flank)
        gate = max(5.0, float(np.percentile(diff, 65)) * 0.45)
        mask = diff > gate
        if not mask.any():
            continue
        padded = np.concatenate(([False], mask, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        if starts.size == 0:
            continue
        runs = ends - starts
        best = int(np.argmax(runs))
        run_start, run_end = int(starts[best]), int(ends[best])
        mid = 0.5 * (run_start + run_end)
        if abs(mid - cw * 0.5) <= cw * 0.22:
            widths.append(run_end - run_start)
    ridge = float(np.median(widths) / cw) if widths else 0.0

    upper = l_eq[: max(4, ch // 3), :]
    col_energy = np.mean(np.abs(cv2.Scharr(upper, cv2.CV_64F, 1, 0)), axis=0)
    mid = cw // 2
    left = col_energy[:mid]
    right = col_energy[mid:]
    bimodality = 0.0
    if left.size and right.size:
        left_peak = float(np.max(left))
        right_peak = float(np.max(right))
        centre = float(np.mean(col_energy[int(cw * 0.44):int(cw * 0.56)])) if cw >= 8 else 0.0
        peak = max(left_peak, right_peak, 1e-6)
        if left_peak > 0.35 * peak and right_peak > 0.35 * peak:
            bimodality = max(0.0, min(1.0, (min(left_peak, right_peak) - centre) / peak))

    return {
        "clahe_vert_edge_score": round(clahe_vert, 3),
        "local_l_contrast": round(local_l, 2),
        "ridge_width_frac": round(ridge, 3),
        "horizontal_bimodality": round(bimodality, 3),
    }


def _infer_neckwear_shape(cv_meta, crop_h, crop_w, face_h):
    """Cheap structural type hint. Never used as a demographic signal.

    The chest crop is almost always wider than it is tall, so crop aspect
    must not be used to declare a bow tie or scarf.
    """
    ridge = float(cv_meta.get("ridge_width_frac") or 0.0)
    bimodality = float(cv_meta.get("horizontal_bimodality") or 0.0)
    short = crop_h <= max(36, int(face_h * 0.55))
    if short and bimodality >= 0.62:
        return "bow_tie"
    if ridge >= 0.07 and ridge <= 0.28:
        return "necktie"
    return ""


def _analyze_tie_cv(bgr, faces):
    """
    Analyze the collar/chest region for structural necktie presence.

    Color and contrast are supporting evidence only. A patterned, multi-colored,
    or shirt-matching blade is still a tie if it occupies the collar-to-placket
    slot as a vertical member. Hard negatives remain: bare throat, V-neck,
    shirt placket, and a monochrome zipper.
    """
    _default_result = {"has_tie": False, "skin_frac": 0.0, "blade_contrast": 0.0, "contrast_ratio": 0.0, "vert_edge_score": 0.0}

    if not faces:
        return {**_default_result, "reason": "no_face"}

    h, w = bgr.shape[:2]
    f = faces[0]
    fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]

    pts = _face_mesh_landmarks(bgr)
    skin_profile = _extract_face_skin_profile(bgr, pts, f)
    skin_lab_mean = skin_profile.get("lab_mean")

    if pts is not None:
        chin = pts[152]
        eye_span = max(10, abs(pts[362][0] - pts[133][0]))
        cx, cy = chin[0], chin[1]
        # Include the collar seat / knot, which sits at or just below the chin.
        cy1 = max(0, cy - int(eye_span * 0.15))
        cy2 = min(h, cy + int(eye_span * 2.80))
        torso_w = max(int(fw * 1.2), int(eye_span * 3.2))
    else:
        cx = fx + fw // 2
        chin_y = min(h, fy + fh)
        cy1 = max(0, chin_y - int(fh * 0.25))
        cy2 = min(h, chin_y + int(fh * 0.95))
        torso_w = int(fw * 1.2)

    cx1 = max(0, cx - torso_w // 2)
    cx2 = min(w, cx + torso_w // 2)

    chest_crop = bgr[cy1:cy2, cx1:cx2]
    if chest_crop.size == 0 or (cy2 - cy1) < 15 or (cx2 - cx1) < 25:
        return {**_default_result, "reason": "insufficient_crop"}

    cw = chest_crop.shape[1]

    # Split the chest crop into three vertical strips: the center (where a tie
    # would be) and the left/right flanks (shirt or jacket fabric).
    c_strip = chest_crop[:, int(cw * 0.36):int(cw * 0.64)]
    l_strip = chest_crop[:, int(cw * 0.05):int(cw * 0.30)]
    r_strip = chest_crop[:, int(cw * 0.70):int(cw * 0.95)]

    # How much visible skin is in the center strip? If we can see a lot of
    # bare skin there, the person probably isn't wearing a tie. We measure
    # closeness to the subject's own skin tone rather than hard-coded values
    # so it works across all skin tones.
    skin_frac = 0.0
    if skin_lab_mean is not None and c_strip.size > 0:
        skin_dists = _compute_skin_distance(c_strip, skin_lab_mean)
        if skin_dists.size > 0:
            is_skin = skin_dists < 12.0
            skin_frac = float(np.mean(is_skin))

    # Compare the average colour of the center strip against the flanks.
    # A tie should look noticeably different from the surrounding shirt.
    c_m = np.mean(c_strip, axis=(0, 1))
    l_m = np.mean(l_strip, axis=(0, 1))
    r_m = np.mean(r_strip, axis=(0, 1))
    flanks_m = (l_m + r_m) / 2.0

    blade_contrast = float(np.linalg.norm(c_m - flanks_m))
    contrast_left = float(np.linalg.norm(c_m - l_m))
    contrast_right = float(np.linalg.norm(c_m - r_m))
    flank_diff = float(np.linalg.norm(l_m - r_m))
    contrast_ratio = blade_contrast / max(10.0, flank_diff * 0.5 + 8.0)

    # Look for vertical edges running along the sides of the tie blade.
    # A real tie creates clear left and right borders against the shirt.
    gray = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2GRAY)
    sob_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    v_left = sob_x[:, int(cw * 0.28):int(cw * 0.42)]
    v_right = sob_x[:, int(cw * 0.58):int(cw * 0.72)]
    vert_edge_score = (float(np.mean(v_left)) + float(np.mean(v_right))) / 2.0 / 15.0

    # Check for texture in the center strip (patterns, stripes, weave).
    # A patterned tie will have higher pixel variance than plain fabric.
    c_gray = gray[:, int(cw * 0.36):int(cw * 0.64)]
    c_var = float(np.std(c_gray))

    # Check whether edges fan out toward the bottom of the crop. A V-neckline
    # gets wider as it goes down; a tie stays roughly the same width.
    ch = chest_crop.shape[0]
    diverging_edges = False
    if ch > 20 and cw > 20:
        upper_third = sob_x[:ch // 3, :]
        lower_third = sob_x[2 * ch // 3:, :]

        # Measure how spread out the edge energy is, top vs. bottom
        def _edge_spread(region):
            col_energy = np.mean(region, axis=0)
            if col_energy.sum() < 1e-6:
                return 0.0
            positions = np.arange(len(col_energy), dtype=np.float32)
            weighted_mean = np.average(positions, weights=col_energy + 1e-9)
            return float(np.sqrt(np.average((positions - weighted_mean) ** 2, weights=col_energy + 1e-9)))

        upper_spread = _edge_spread(upper_third)
        lower_spread = _edge_spread(lower_third)
        # If the bottom edges are much wider than the top ones → V-neckline
        if lower_spread > upper_spread * 1.4 and lower_spread > 8.0:
            diverging_edges = True

    # A tie should create roughly equal contrast on both sides of the center.
    # Things like zippers, collar folds, and button lines tend to be lopsided.
    bilateral_symmetry = 1.0 - abs(contrast_left - contrast_right) / max(1.0, max(contrast_left, contrast_right))

    # If the center strip is basically the same colour as the flanks (e.g. a
    # zipper on a solid-colour jacket), there's no distinct tie — whatever
    # contrast we picked up is just texture, not a different-coloured tie.
    c_lab = cv2.cvtColor(c_strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    c_chroma = float(np.mean(np.sqrt(c_lab[:, :, 1] ** 2 + c_lab[:, :, 2] ** 2)))
    l_lab = cv2.cvtColor(l_strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(r_strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    flanks_chroma = (float(np.mean(np.sqrt(l_lab[:, :, 1] ** 2 + l_lab[:, :, 2] ** 2))) +
                     float(np.mean(np.sqrt(r_lab[:, :, 1] ** 2 + r_lab[:, :, 2] ** 2)))) / 2.0
    chroma_diff = abs(c_chroma - flanks_chroma)

    robust = _illumination_robust_tie_cues(chest_crop)
    necktie_ridge = 0.07 <= float(robust["ridge_width_frac"]) <= 0.28
    residual_structure = (
        float(robust["clahe_vert_edge_score"]) >= 0.28
        or float(robust["local_l_contrast"]) >= 3.5
        or necktie_ridge
    )

    # Pack up all the measurements so we can include them in the response.
    result_meta = {
        "skin_frac": round(skin_frac, 3),
        "blade_contrast": round(blade_contrast, 2),
        "contrast_ratio": round(contrast_ratio, 2),
        "vert_edge_score": round(vert_edge_score, 2),
        "bilateral_symmetry": round(bilateral_symmetry, 3),
        "chroma_diff": round(chroma_diff, 2),
        "center_variance": round(c_var, 2),
        "clahe_vert_edge_score": robust["clahe_vert_edge_score"],
        "local_l_contrast": robust["local_l_contrast"],
        "ridge_width_frac": robust["ridge_width_frac"],
        "horizontal_bimodality": robust["horizontal_bimodality"],
        "residual_structure": bool(residual_structure),
        "knot_only_crop": bool(ch <= max(36, int(fh * 0.55))),
    }

    # A tie blade has fairly even brightness from top to bottom. If there's
    # a big light-to-dark jump, it's more likely a V-neckline showing an
    # undershirt underneath.
    c_lab_l = c_lab[:, :, 0]
    c_rows = c_lab_l.shape[0]
    vert_gradient = 0.0
    if c_rows >= 6:
        top_L = float(np.mean(c_lab_l[:c_rows // 3, :]))
        bot_L = float(np.mean(c_lab_l[2 * c_rows // 3:, :]))
        vert_gradient = abs(top_L - bot_L)
    result_meta["vert_gradient"] = round(vert_gradient, 2)

    # Too much visible skin in the center → no tie
    if skin_frac > 0.18:
        return {**result_meta, "has_tie": False, "reason": "bare_skin"}

    # Plain shirt with nothing going on in the center → no tie
    max_flank_contrast = max(contrast_left, contrast_right)
    if (
        blade_contrast < 25.0
        and max_flank_contrast < 28.0
        and vert_edge_score < 0.40
        and c_var < 18.0
        and float(robust["clahe_vert_edge_score"]) < 0.45
        and not necktie_ridge
    ):
        return {**result_meta, "has_tie": False, "reason": "solid_shirt_no_tie"}

    # Edges fan outward (V-neckline or open collar) → not a tie
    if diverging_edges and blade_contrast < 50.0:
        return {**result_meta, "has_tie": False, "reason": "v_neckline"}

    # Center looks the same colour as the flanks (zipper, seam, etc.) → not a tie
    # unless illumination-robust edges show a necktie-width ridge. A zipper is
    # thinner than that ridge and still fails this gate.
    if (
        chroma_diff < 4.0
        and blade_contrast < 45.0
        and c_var < 22.0
        and not (
            float(robust["clahe_vert_edge_score"]) >= 0.50
            and necktie_ridge
            and float(robust["local_l_contrast"]) >= 4.0
        )
    ):
        return {**result_meta, "has_tie": False, "reason": "monochrome_garment_feature"}

    # Lopsided contrast is typical of a button line or collar fold, but a
    # patterned / multi-colored blade is also often left-right uneven. Only
    # reject when the center is bland *and* not chromatically distinct.
    if (
        bilateral_symmetry < 0.45
        and blade_contrast < 50.0
        and c_var < 22.0
        and chroma_diff < 3.5
    ):
        return {**result_meta, "has_tie": False, "reason": "asymmetric_contrast"}

    # Big brightness swing + high texture variance → open collar / undershirt,
    # not a necktie blade. Do not let high mean contrast override this: a
    # white undershirt against a dark shirt is high-contrast and still not a tie.
    if vert_gradient > 50.0 and c_var > 30.0:
        return {**result_meta, "has_tie": False, "reason": "v_neckline_luminance_gradient"}

    # Classic high-contrast solid tie.
    classic_tie = (
        bilateral_symmetry >= 0.45
        and vert_gradient < 50.0
        and (
            (blade_contrast >= 30.0 and vert_edge_score >= 0.45 and chroma_diff >= 3.0)
            or (max_flank_contrast >= 45.0 and vert_edge_score >= 0.45)
            or (
                blade_contrast >= 30.0
                and c_var >= 28.0
                and chroma_diff >= 3.0
            )
            or (blade_contrast >= 50.0 and contrast_ratio >= 1.30)
        )
    )
    # Patterned / multi-colored / paisley blade: the center column is a
    # distinct vertical member. Do not require left/right colour symmetry
    # or a single dominant hue. Fine weaves can have modest pixel variance.
    patterned_tie = (
        blade_contrast >= 18.0
        and chroma_diff >= 3.5
        and vert_gradient < 40.0
        and (c_var >= 14.0 or blade_contrast >= 35.0)
    )
    # Dark-on-dark / low-contrast: a vertical member with flanking edges,
    # even when mean colour matches the shirt. CLAHE-on-L is analysis-only.
    low_contrast_tie = (
        (
            vert_edge_score >= 0.55
            and bilateral_symmetry >= 0.40
            and blade_contrast >= 8.0
            and vert_gradient < 50.0
        )
        or (
            float(robust["clahe_vert_edge_score"]) >= 0.50
            and necktie_ridge
            and bilateral_symmetry >= 0.35
            and float(robust["local_l_contrast"]) >= 3.5
            and vert_gradient < 55.0
        )
    )
    # Upper knot only: the frame cuts through the collar, so there is no
    # blade to measure. A compact, high-contrast mass in the collar seat
    # is enough. Do not require chroma, symmetry, or a tall crop.
    short_collar_crop = ch <= max(36, int(fh * 0.55))
    knot_only_tie = (
        short_collar_crop
        and vert_gradient < 45.0
        and skin_frac <= 0.15
        and (
            blade_contrast >= 22.0
            or (
                float(robust["local_l_contrast"]) >= 5.0
                and float(robust["clahe_vert_edge_score"]) >= 0.40
            )
        )
    )

    is_tie = (
        skin_frac <= 0.15
        and not diverging_edges
        and (classic_tie or patterned_tie or low_contrast_tie or knot_only_tie)
    )
    if is_tie and knot_only_tie and not (classic_tie or patterned_tie or low_contrast_tie):
        reason = "knot_only_detected"
    elif is_tie and patterned_tie and not classic_tie:
        reason = "patterned_tie_detected"
    elif is_tie and low_contrast_tie and not classic_tie:
        reason = "low_contrast_tie_detected"
    elif is_tie:
        reason = "tie_detected"
    else:
        reason = "insufficient_tie_profile"

    result_meta["neckwear_shape"] = _infer_neckwear_shape(
        result_meta, ch, cw, fh
    ) if is_tie else ""
    return {
        **result_meta,
        "has_tie": bool(is_tie),
        "reason": reason,
    }


def _structural_tie_recovery(bgr, faces, model_version, **detector_meta):
    """Accept a required tie from collar/chest structure when the box model misses.

    COCO and similar detectors are biased toward solid dark ties. A patterned
    or low-contrast necktie can be structurally obvious while producing no box.
    That miss is not proof of absence.
    """
    cv_res = detector_meta.pop("cv_result", None)
    if cv_res is None:
        cv_res = _analyze_tie_cv(bgr, faces)
    if not cv_res.get("has_tie"):
        return None
    meta = {
        "decision": "accept",
        "tie_status": "tie_present",
        "tie_detected": True,
        "confidence": round(float(detector_meta.get("confidence") or 0.0), 4),
        "upper_body_visible": True,
        "model_version": model_version,
        "reason": "structural_cv",
        "cv_reason": cv_res.get("reason"),
        "corroborated_by_cv": True,
    }
    bbox = detector_meta.get("bbox")
    if bbox is not None:
        meta["bbox"] = bbox
    return {
        "passed": True,
        "message": "Tie / formal neckwear detected.",
        "meta": meta,
    }


def _tie_runtime_cache(params):
    """Return request-local tie evidence storage.

    ``run_checks`` executes criteria sequentially. Keeping this cache in the
    request's params object avoids a second Faster R-CNN pass when both tie
    criteria are enabled, while direct callers remain fully supported.
    """
    if not isinstance(params, dict):
        return None
    return params.setdefault("_runtime_cache", {})


def _cached_tie_inference(bgr, roi, params):
    """Run tie-model inference once for the current image/ROI."""
    cache = _tie_runtime_cache(params)
    key = ("tie_inference", id(bgr), tuple(roi))
    if cache is not None and key in cache:
        return cache[key]

    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_bgr = bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    try:
        detector = get_tie_detector()
        detection = detector.detect(
            Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)),
            roi_offset=(roi_x1, roi_y1),
        )
        value = (detector, detection, None)
    except Exception as exc:
        _logger.warning("Tie detector unavailable: %s", exc)
        value = (None, None, exc)

    if cache is not None:
        cache[key] = value
    return value


def _cached_structural_tie_analysis(bgr, faces, params):
    """Share the CPU-only structural tie fallback across tie criteria."""
    cache = _tie_runtime_cache(params)
    key = ("tie_structural_cv", id(bgr))
    if cache is not None and key in cache:
        return cache[key]
    value = _analyze_tie_cv(bgr, faces)
    if cache is not None:
        cache[key] = value
    return value


def check_tie(bgr, faces, params):
    """Verify a required official necktie using staged evidence.

    A one-class object detector can establish *presence* but cannot establish
    *absence*: a missing box could be an open collar, a crop, occlusion, or a
    model miss. Bow ties, scarves, and other non-necktie neckwear never auto-
    pass. A missing official-design specification cannot be invented; default
    enforcement is type-only (traditional necktie). Weak or cropped evidence
    is reviewed, not reported as “no tie”.
    """
    model_version = os.environ.get("TIE_MODEL_VERSION", "tie-detector-dev")
    if not faces:
        return _tie_manual_review(
            "Face not detected — cannot evaluate tie presence.",
            status="insufficient_upper_body_visibility", visible=False,
            model_version=model_version,
        )

    face = faces[0]
    h, w = bgr.shape[:2]
    min_vis_ratio = params.get("tie_min_visible_below_face_ratio")
    min_face_h = params.get("tie_min_face_height")
    estimator = (
        UpperBodyVisibilityEstimator(
            min_face_height_px=int(min_face_h) if min_face_h is not None else None,
            min_visible_below_face_ratio=float(min_vis_ratio) if min_vis_ratio is not None else None,
        )
        if min_vis_ratio is not None or min_face_h is not None else _upper_body_estimator
    )
    vis = estimator.estimate(face, w, h)
    if not vis.sufficient:
        return _tie_manual_review(
            "The neck/chest region is not sufficiently visible to determine whether formal neckwear is present.",
            status="insufficient_upper_body_visibility", visible=False,
            model_version=model_version, visibility_reason=vis.reason,
        )

    detector, detection, detector_error = _cached_tie_inference(
        bgr, vis.roi, params
    )
    policy = load_official_tie_policy(params)
    cv_res = _cached_structural_tie_analysis(bgr, faces, params)

    if detector_error is not None:
        fallback_type = classify_neckwear_type(
            cv_res=cv_res,
            bbox=None,
            face=face,
            image_hw=(h, w),
            visibility_reason=vis.reason,
            detection_valid=False,
        )
        return decide_require_tie(
            policy=policy,
            cv_res=cv_res,
            detection_valid=False,
            detection_confidence=0.0,
            threshold=0.65,
            localization_reason=None,
            visibility_reason=vis.reason,
            visibility_sufficient=True,
            neckwear_type=fallback_type,
            official_match="unspecified",
            supports_absence=False,
            model_version=model_version,
            detector_available=False,
        )

    threshold, geometry, model_version = _tie_positive_policy(detector, params)
    bbox = None
    bbox_tuple = None
    detection_valid = False
    localization_reason = None
    confidence = 0.0
    if detection is not None:
        confidence = float(detection.confidence)
        bbox_tuple = tuple(detection.bbox)
        bbox = {key: int(value) for key, value in zip(("x1", "y1", "x2", "y2"), detection.bbox)}
        detection_valid, localization_reason = validate_tie_detection(
            detection, face, w, h, geometry
        )
    else:
        bbox_tuple = None

    neckwear_type = classify_neckwear_type(
        cv_res=cv_res,
        bbox=bbox_tuple,
        face=face,
        image_hw=(h, w),
        visibility_reason=vis.reason,
        detection_valid=detection_valid,
    )
    appearance_stats = measure_appearance(bgr, bbox_tuple)
    official_match = match_official_appearance(appearance_stats, policy)
    if official_match == "mismatch" and neckwear_type == "necktie":
        neckwear_type = "unofficial_tie"

    return decide_require_tie(
        policy=policy,
        cv_res=cv_res,
        detection_valid=bool(detection_valid),
        detection_confidence=confidence,
        threshold=threshold,
        localization_reason=localization_reason if detection is not None else None,
        visibility_reason=vis.reason,
        visibility_sufficient=True,
        neckwear_type=neckwear_type,
        official_match=official_match,
        supports_absence=getattr(detector, "supports_absence_decision", False) is True,
        model_version=model_version,
        bbox=bbox,
        detector_available=True,
    )


# ---------- Learned Tie Detector (no_tie criterion) ----------


def check_no_tie(bgr, faces, params):
    """Learned necktie detector with upper-body visibility gate.

    Returns one of four states:
        tie_present                     -> passed=False (reject)
        tie_absent + visible            -> passed=True  (accept)
        uncertain                       -> passed=False (manual_review)
        insufficient_upper_body_visibility -> passed=False (manual_review)

    The detector uses the trained model when available, and falls back to
    robust computer-vision necktie analysis.
    """
    # Confidence thresholds (from env / params)
    reject_threshold = float(
        params.get("tie_reject_threshold",
                   os.environ.get("TIE_REJECT_THRESHOLD", "0.65"))
    )
    accept_threshold = float(
        params.get("tie_accept_threshold",
                   os.environ.get("TIE_ACCEPT_THRESHOLD", "0.30"))
    )
    model_version = os.environ.get("TIE_MODEL_VERSION", "tie-detector-dev")

    # --- No face: cannot determine tie status ---
    if not faces:
        return {
            "passed": False,
            "message": "Face not detected — cannot evaluate tie presence.",
            "meta": {
                "decision": "manual_review",
                "tie_status": "insufficient_upper_body_visibility",
                "tie_detected": None,
                "upper_body_visible": False,
                "model_version": model_version,
            },
        }

    face = faces[0]
    h, w = bgr.shape[:2]

    # --- Visibility gate ---
    min_vis_ratio = params.get("tie_min_visible_below_face_ratio")
    min_face_h = params.get("tie_min_face_height")
    if min_vis_ratio is not None or min_face_h is not None:
        estimator = UpperBodyVisibilityEstimator(
            min_face_height_px=int(min_face_h) if min_face_h is not None else None,
            min_visible_below_face_ratio=float(min_vis_ratio) if min_vis_ratio is not None else None,
        )
        vis = estimator.estimate(face, w, h)
    else:
        vis = _upper_body_estimator.estimate(face, w, h)
    if not vis.sufficient:
        return {
            "passed": False,
            "message": (
                "The neck/chest region is not sufficiently visible "
                "to determine whether a tie is present."
            ),
            "meta": {
                "decision": "manual_review",
                "tie_status": "insufficient_upper_body_visibility",
                "tie_detected": None,
                "upper_body_visible": False,
                "visibility_reason": vis.reason,
                "model_version": model_version,
            },
        }

    # --- Attempt to load the learned detector ---
    detector, detection, detector_error = _cached_tie_inference(
        bgr, vis.roi, params
    )
    if detector_error is not None:
        return {
            "passed": False,
            "message": (
                "Tie detection model is not available. "
                "Manual review is required."
            ),
            "meta": {
                "decision": "manual_review",
                "tie_status": "uncertain",
                "tie_detected": None,
                "upper_body_visible": True,
                "error": "model_unavailable",
                "model_version": model_version,
            },
        }

    # Apply the same face-relative localization guard used by require_tie.
    # A high detector score on a lapel, necklace, background pattern, or a
    # second person must never be treated as a visible tie.
    calibrated_threshold, geometry, policy_version = _tie_positive_policy(detector, params)
    if isinstance(getattr(detector, "policy", None), TieModelPolicy):
        reject_threshold = calibrated_threshold
        model_version = policy_version

    if detection is None:
        cv_res = _cached_structural_tie_analysis(bgr, faces, params)
        if cv_res.get("has_tie"):
            return {
                "passed": False,
                "message": "Visible necktie detected in the upper-body region.",
                "meta": {
                    "decision": "reject",
                    "tie_status": "tie_present",
                    "tie_detected": True,
                    "confidence": 0.0,
                    "upper_body_visible": True,
                    "corroborated_by_cv": True,
                    "cv_reason": cv_res.get("reason"),
                    "model_version": model_version,
                },
            }
        if getattr(detector, "supports_absence_decision", False) is not True:
            return _tie_manual_review(
                "No reliable tie absence decision was produced. Manual review is required.",
                status="uncertain", visible=True, model_version=model_version,
                reason="absence_not_supported", confidence=0.0,
            )
        return {
            "passed": True,
            "message": (
                "No necktie detected and sufficient upper-body "
                "visibility is available."
            ),
            "meta": {
                "decision": "accept",
                "tie_status": "tie_absent",
                "tie_detected": False,
                "confidence": 1.0,
                "upper_body_visible": True,
                "model_version": model_version,
            },
        }

    conf = detection.confidence
    bbox_dict = {
        "x1": int(detection.bbox[0]),
        "y1": int(detection.bbox[1]),
        "x2": int(detection.bbox[2]),
        "y2": int(detection.bbox[3]),
    }

    valid, localization_reason = validate_tie_detection(detection, face, w, h, geometry)
    if not valid:
        return _tie_manual_review(
            "Tie-like object is outside the expected neck/chest region. Manual review is required.",
            status="uncertain", visible=True, model_version=model_version,
            reason=localization_reason, confidence=round(detection.confidence, 4),
            bbox=bbox_dict,
        )

    if conf >= reject_threshold:
        return {
            "passed": False,
            "message": "Visible necktie detected in the upper-body region.",
            "meta": {
                "decision": "reject",
                "tie_status": "tie_present",
                "tie_detected": True,
                "confidence": round(conf, 4),
                "upper_body_visible": True,
                "bbox": bbox_dict,
                "model_version": model_version,
            },
        }

    if conf <= accept_threshold:
        cv_res = _cached_structural_tie_analysis(bgr, faces, params)
        if cv_res.get("has_tie"):
            return {
                "passed": False,
                "message": "Visible necktie detected in the upper-body region.",
                "meta": {
                    "decision": "reject",
                    "tie_status": "tie_present",
                    "tie_detected": True,
                    "confidence": round(conf, 4),
                    "upper_body_visible": True,
                    "bbox": bbox_dict,
                    "corroborated_by_cv": True,
                    "cv_reason": cv_res.get("reason"),
                    "model_version": model_version,
                },
            }
        return {
            "passed": True,
            "message": (
                "No necktie detected and sufficient upper-body "
                "visibility is available."
            ),
            "meta": {
                "decision": "accept",
                "tie_status": "tie_absent",
                "tie_detected": False,
                "confidence": round(conf, 4),
                "upper_body_visible": True,
                "bbox": bbox_dict,
                "model_version": model_version,
            },
        }

    # --- Uncertain band: cross-check with CV heuristic for corroboration ---
    cv_res = _cached_structural_tie_analysis(bgr, faces, params)
    if cv_res["has_tie"]:
        # Both signals agree: tie is present → reject
        return {
            "passed": False,
            "message": "Visible necktie detected in the upper-body region.",
            "meta": {
                "decision": "reject",
                "tie_status": "tie_present",
                "tie_detected": True,
                "confidence": round(conf, 4),
                "upper_body_visible": True,
                "bbox": bbox_dict,
                "corroborated_by_cv": True,
                "model_version": model_version,
            },
        }
    # CV disagrees (no tie) while model is uncertain → manual review
    return {
        "passed": False,
        "message": (
            "Tie presence is uncertain in the visible upper-body "
            "region. Manual review is required."
        ),
        "meta": {
            "decision": "manual_review",
            "tie_status": "uncertain",
            "tie_detected": None,
            "confidence": round(conf, 4),
            "upper_body_visible": True,
            "bbox": bbox_dict,
            "model_version": model_version,
        },
    }





def check_resolution(bgr, faces, params):
    h, w = bgr.shape[:2]
    min_w = params.get("min_width", 400)
    min_h = params.get("min_height", 500)
    if w < min_w or h < min_h:
        return {"passed": False,
                "message": f"Image resolution too low ({w}x{h}). Minimum required is {min_w}x{min_h}.",
                "meta": {"width": w, "height": h}}
    return {"passed": True, "message": f"Resolution OK ({w}x{h}).", "meta": {"width": w, "height": h}}


def check_passport_size_ratio(bgr, faces, params):
    """Standard passport photo aspect ratio check, e.g. 2:2 (square) or 35mm:45mm (~0.777)."""
    h, w = bgr.shape[:2]
    ratio = w / float(h)
    target = params.get("target_ratio", 0.777)  # 35x45mm default
    tolerance = params.get("ratio_tolerance", 0.12)
    if abs(ratio - target) > tolerance:
        return {"passed": False,
                "message": f"Photo aspect ratio ({ratio:.2f}) does not match required passport size ratio ({target:.2f}).",
                "meta": {"ratio": ratio}}
    return {"passed": True, "message": "Aspect ratio matches passport size standard.", "meta": {"ratio": ratio}}


def check_blur(bgr, faces, params):
    """
    Tiered sharpness detection with separate severity levels.

    Instead of a single binary pass/fail, this check classifies sharpness into
    three tiers so that downstream logic (``run_checks``) can distinguish
    between an acceptably sharp image, a moderately blurred but still usable
    image (common on mobile cameras), and a severely blurred image that
    prevents reliable validation.

    Severity tiers (reported in ``meta.severity``):
    - ``acceptable``: sharpness >= blur_threshold — passes unconditionally.
    - ``soft``:       blur_severe_threshold <= sharpness < blur_threshold —
                      below the ideal threshold but the image is still usable.
                      When ``blur_soft_fail`` is enabled (default), this is
                      treated as a soft failure that does not block overall
                      approval when the primary compliance check (white
                      background) independently passes.
    - ``severe``:     sharpness < blur_severe_threshold — the image is too
                      blurred for reliable validation and always fails.
    """
    variance = _measure_sharpness(bgr, faces)

    threshold = float(params.get("blur_threshold", 80.0))
    severe_threshold = float(params.get("blur_severe_threshold", 15.0))
    soft_fail_enabled = bool(int(params.get("blur_soft_fail", 1)))

    meta = {
        "sharpness": variance,
        "blur_threshold": threshold,
        "blur_severe_threshold": severe_threshold,
        "blur_soft_fail_enabled": soft_fail_enabled,
    }

    if variance >= threshold:
        meta["severity"] = "acceptable"
        return {"passed": True, "message": "Photo sharpness is acceptable.", "meta": meta}

    if variance < severe_threshold:
        meta["severity"] = "severe"
        return {
            "passed": False,
            "message": "Photo is severely blurred. Image quality is insufficient for reliable verification. Please retake with a steady camera and good focus.",
            "meta": meta,
        }

    # Soft blur range: below ideal threshold but above severe cutoff
    meta["severity"] = "soft"
    if soft_fail_enabled:
        # Mark as failed for the individual check, but run_checks() will
        # treat this as a conditional pass when white_background passes.
        return {
            "passed": False,
            "message": "Photo sharpness is below the ideal threshold but may still be acceptable if other quality requirements are met.",
            "meta": meta,
        }
    # When soft-fail is disabled by the admin, any blur below threshold fails.
    return {
        "passed": False,
        "message": "Photo appears blurry. Please retake with a steady camera and good focus.",
        "meta": meta,
    }


def check_brightness(bgr, faces, params):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    low = params.get("min_brightness", 60)
    high = params.get("max_brightness", 230)
    if mean_brightness < low:
        return {"passed": False, "message": "Photo is too dark. Please retake in better lighting.",
                "meta": {"brightness": mean_brightness}}
    if mean_brightness > high:
        return {"passed": False, "message": "Photo is overexposed / too bright.",
                "meta": {"brightness": mean_brightness}}
    return {"passed": True, "message": "Lighting/exposure is acceptable.", "meta": {"brightness": mean_brightness}}


def check_head_pose(bgr, faces, params):
    """Uses face mesh to estimate whether head is tilted/turned too much."""
    pts = _face_mesh_landmarks(bgr)
    if pts is None:
        return {
            "passed": False,
            "message": "Head pose could not be verified. Manual review is required.",
            "meta": {"decision": "manual_review", "error": "landmarks_unavailable"},
        }

    # Landmark indices: left eye outer (33), right eye outer (263), nose tip (1), chin (152)
    try:
        left_eye = np.array(pts[33])
        right_eye = np.array(pts[263])
        nose = np.array(pts[1])
        chin = np.array(pts[152])
    except IndexError:
        return {
            "passed": False,
            "message": "Head pose could not be verified. Manual review is required.",
            "meta": {"decision": "manual_review", "error": "landmarks_incomplete"},
        }

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll_angle = np.degrees(np.arctan2(dy, dx))

    eye_mid = (left_eye + right_eye) / 2.0
    face_vertical = chin - eye_mid
    horizontal_offset = nose[0] - eye_mid[0]
    yaw_estimate = horizontal_offset / (np.linalg.norm(right_eye - left_eye) + 1e-6)

    max_roll = params.get("max_roll_deg", 12)
    max_yaw = params.get("max_yaw_ratio", 0.35)

    issues = []
    if abs(roll_angle) > max_roll:
        issues.append("Head is tilted. Please face the camera straight-on.")
    if abs(yaw_estimate) > max_yaw:
        issues.append("Face is turned to the side. Please look directly at the camera.")

    if issues:
        return {"passed": False, "message": " ".join(issues),
                "meta": {"roll": float(roll_angle), "yaw_ratio": float(yaw_estimate)}}
    return {"passed": True, "message": "Head pose is straight and centered.",
            "meta": {"roll": float(roll_angle), "yaw_ratio": float(yaw_estimate)}}


def check_eyes_open(bgr, faces, params):
    pts = _face_mesh_landmarks(bgr)
    if pts is None:
        return {
            "passed": False,
            "message": "Eyes could not be verified. Manual review is required.",
            "meta": {"decision": "manual_review", "error": "landmarks_unavailable"},
        }
    try:
        # Left eye vertical landmarks: 159 (top), 145 (bottom); horizontal: 33,133
        top = np.array(pts[159]); bottom = np.array(pts[145])
        left = np.array(pts[33]); right = np.array(pts[133])
        vert = np.linalg.norm(top - bottom)
        horiz = np.linalg.norm(left - right) + 1e-6
        ear = vert / horiz
    except IndexError:
        return {
            "passed": False,
            "message": "Eyes could not be verified. Manual review is required.",
            "meta": {"decision": "manual_review", "error": "landmarks_incomplete"},
        }

    threshold = params.get("eye_open_ratio", 0.15)
    if ear < threshold:
        return {"passed": False, "message": "Eyes appear closed. Please keep your eyes open and retake the photo.",
                "meta": {"ear": float(ear)}}
    return {"passed": True, "message": "Eyes are open.", "meta": {"ear": float(ear)}}


# ---------- Registry ----------

CHECK_REGISTRY = {
    "single_face": check_face_count,
    "face_framing": check_face_size_centering,
    "no_glasses": check_glasses,
    "white_background": check_white_background,
    "require_tie": check_tie,
    "no_tie": check_no_tie,
    "min_resolution": check_resolution,
    "passport_ratio": check_passport_size_ratio,
    "no_blur": check_blur,
    "brightness": check_brightness,
    "head_pose": check_head_pose,
    "eyes_open": check_eyes_open,
}

CHECK_LABELS = {
    "single_face": "Single Face Detection",
    "face_framing": "Face Size & Centering",
    "no_glasses": "No Eyeglasses",
    "white_background": "Strictly White Background",
    "require_tie": "Tie / Formal Neckwear Required",
    "no_tie": "No Necktie",
    "min_resolution": "Minimum Resolution",
    "passport_ratio": "Passport Size Aspect Ratio",
    "no_blur": "Sharpness (No Blur)",
    "brightness": "Proper Lighting / Exposure",
    "head_pose": "Straight Head Pose",
    "eyes_open": "Eyes Open",
}

# Every criterion enabled by the university policy is a requirement for an
# automatic admissions-photo pass. A pass-count threshold must not hide a
# failed face, framing, quality, or lighting check.
_MANDATORY_CRITERIA = frozenset(CHECK_REGISTRY)


def _is_mandatory(key):
    """True when a failed criterion must always block the overall verdict."""
    return key in _MANDATORY_CRITERIA


def run_checks(image_bytes, enabled_criteria, params=None):
    """
    enabled_criteria: dict {criteria_key: bool}
    params: dict of tunable numeric parameters (optional, merges with defaults)

    All-enabled-criteria policy
    ----------------------------
    Every enabled criterion is a hard requirement for automatic approval.
    ``min_pass_criteria`` and the legacy ``strict_all_criteria`` parameter are
    retained for request/response compatibility, but cannot override a failed
    enabled requirement.
    """
    # Keep runtime evidence (including the tie cache) isolated to this request;
    # callers may reuse their configuration dictionary across submissions.
    params = dict(params or {})
    started_at = time.perf_counter()
    try:
        bgr = _load_image(image_bytes)
    except ImageValidationError as exc:
        return {
            "overall_passed": False,
            "error": str(exc),
            "results": {},
        }
    except Exception:
        # Never surface decoder internals to the caller.
        _logger.exception("Unexpected failure while decoding uploaded image.")
        return {
            "overall_passed": False,
            "error": "Could not read image.",
            "results": {},
        }

    decoded_at = time.perf_counter()
    faces = _detect_faces(bgr)
    faces_detected_at = time.perf_counter()

    results = {}
    overall_passed = True

    # Always run single_face first if enabled, others depend on face presence but degrade gracefully
    ordered_keys = [k for k in CHECK_REGISTRY if enabled_criteria.get(k)]

    passed_count = 0
    check_timings_ms = {}
    for key in ordered_keys:
        fn = CHECK_REGISTRY[key]
        check_started_at = time.perf_counter()
        try:
            res = fn(bgr, faces, params)
        except Exception:
            # A single failing check must not abort the whole verification,
            # but its internals must not reach the student-facing response.
            _logger.exception("Check '%s' raised an unexpected error.", key)
            res = {
                "passed": False,
                "message": "This check could not be completed. Manual review is required.",
                "meta": {"error": "check_failed"},
            }
        check_timings_ms[key] = round((time.perf_counter() - check_started_at) * 1000.0, 3)
        res["label"] = CHECK_LABELS.get(key, key)
        results[key] = res
        if res.get("passed"):
            passed_count += 1

    # Report soft blur separately, but never turn a failed enabled quality
    # requirement into a pass.
    quality_notes = []
    blur_result = results.get("no_blur", {})
    bg_result = results.get("white_background", {})

    if (
        enabled_criteria.get("no_blur")
        and not blur_result.get("passed")
        and blur_result.get("meta", {}).get("severity") == "soft"
        and blur_result.get("meta", {}).get("blur_soft_fail_enabled")
    ):
        bg_passed = bg_result.get("passed", True) if enabled_criteria.get("white_background") else True
        if bg_passed:
            quality_notes.append(
                "Sharpness is below the ideal threshold; manual review or retake is required."
            )

    total_criteria = len(ordered_keys)
    # Legacy pass-count/strictness inputs remain accepted but are no longer
    # allowed to approve a photo that failed an enabled criterion.
    strict_mode = True

    # No failure is maskable under the admissions policy.
    masked_failures = [
        key for key, res in results.items()
        if enabled_criteria.get(key) and not res.get("passed")
    ]

    if total_criteria == 0:
        overall_passed = True
        required_to_pass = 0
    else:
        required_to_pass = total_criteria
        # Exact-one-face is a prerequisite even if a caller accidentally omits
        # the single_face criterion from its enabled map.
        overall_passed = passed_count == total_criteria and len(faces) == 1

    completed_at = time.perf_counter()
    response = {
        "overall_passed": overall_passed,
        "passed_count": passed_count,
        "total_criteria": total_criteria,
        "required_to_pass": required_to_pass,
        "masked_failures": [],
        "failed_criteria": masked_failures,
        "results": results,
        "image_info": {
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
            "faces_detected": len(faces)
        },
        "timings_ms": {
            "image_decode": round((decoded_at - started_at) * 1000.0, 3),
            "face_detection": round((faces_detected_at - decoded_at) * 1000.0, 3),
            "checks": check_timings_ms,
            "total": round((completed_at - started_at) * 1000.0, 3),
        },
    }
    if quality_notes:
        response["quality_notes"] = quality_notes
    return response
