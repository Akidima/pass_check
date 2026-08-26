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
import os
import threading
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
from verify import run_checks, CHECK_LABELS, _detect_faces

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
CORS(app)

MAX_UPLOAD_MB = 12
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def get_rembg_session():
    """Load the optional U2-Net session once, only when background editing needs it."""
    global REMBG_SESSION, _REMBG_INIT_ATTEMPTED
    if not REMBG_AVAILABLE:
        return None
    with _REMBG_LOCK:
        if not _REMBG_INIT_ATTEMPTED:
            _REMBG_INIT_ATTEMPTED = True
            try:
                REMBG_SESSION = rembg_new_session('u2net')
            except Exception:
                REMBG_SESSION = None
    return REMBG_SESSION


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
            cutout_img = rembg_remove(pil_img_rgb, session=rembg_session)
        except Exception as e:
            print(f"rembg_remove error: {e}")
            pass

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

    # Locate subject head center for dynamic background alignment
    center_x = w * 0.5
    center_y = h * 0.38
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

    is_white_bg = (target_rgb[0] >= 240 and target_rgb[1] >= 240 and target_rgb[2] >= 240)
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
    pil_img = Image.open(io.BytesIO(image_bytes))
    try:
        from PIL import ImageOps
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    pil_img = pil_img.convert("RGB")

    # 1. Background Color Replacement with dynamic studio lighting & light wrap
    if bg_color:
        pil_img = replace_background_color(pil_img, bg_color)

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

    # 4. Resolution / Resizing
    if width and height:
        try:
            w = int(width)
            h = int(height)
            if w > 0 and h > 0:
                pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        except (ValueError, TypeError):
            pass

    out_buf = io.BytesIO()
    pil_img.save(out_buf, format="JPEG", quality=96, subsampling=0)
    return out_buf.getvalue()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "passport-verification", "checks_available": list(CHECK_LABELS.keys())})


@app.route("/verify", methods=["POST"])
def verify():
    if "photo" not in request.files:
        return jsonify({"error": "No photo file provided (field name must be 'photo')."}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty file."}), 400

    raw_criteria = request.form.get("criteria", "{}")
    raw_params = request.form.get("params", "{}")

    try:
        enabled_criteria = json.loads(raw_criteria)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid 'criteria' JSON."}), 400

    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError:
        params = {}

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
    status_code = 200 if "error" not in result else 422
    return jsonify(result), status_code


@app.route("/edit-photo", methods=["POST"])
def edit_photo():
    if "photo" not in request.files:
        return jsonify({"error": "No photo file provided."}), 400

    file = request.files["photo"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Empty photo file."}), 400

    get_cutout = request.form.get("get_cutout", "false").lower() in ("true", "1")
    if get_cutout:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            try:
                from PIL import ImageOps
                pil_img = ImageOps.exif_transpose(pil_img)
            except Exception:
                pass
            cutout_img = get_subject_cutout(pil_img)
            out_buf = io.BytesIO()
            cutout_img.save(out_buf, format="PNG")
            return send_file(io.BytesIO(out_buf.getvalue()), mimetype="image/png")
        except Exception as e:
            return jsonify({"error": f"Failed to generate cutout: {str(e)}"}), 500

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
    except Exception as e:
        return jsonify({"error": f"Failed to process photo edit: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="127.0.0.1", port=port, debug=True)
