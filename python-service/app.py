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
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from verify import run_checks, CHECK_LABELS, _detect_faces

try:
    from rembg import new_session as rembg_new_session, remove as rembg_remove
    REMBG_SESSION = rembg_new_session('u2net')
    REMBG_AVAILABLE = True
except Exception:
    REMBG_SESSION = None
    REMBG_AVAILABLE = False

app = Flask(__name__)
CORS(app)

MAX_UPLOAD_MB = 12
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def get_subject_cutout(pil_img):
    """
    Isolates the subject/person completely using U2-Net deep learning portrait segmentation (rembg)
    or OpenCV GrabCut/FloodFill fallback, returning a PIL Image in RGBA mode with a transparent background.
    """
    pil_img_rgb = pil_img.convert("RGB")

    if REMBG_AVAILABLE and REMBG_SESSION is not None:
        try:
            return rembg_remove(pil_img_rgb, session=REMBG_SESSION)
        except Exception as e:
            print(f"rembg_remove error: {e}")
            pass

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

        if faces:
            f = faces[0]
            fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]
            body_top = min(h - 1, fy + int(fh * 0.75))
            final_bg_mask[body_top:h, :] = 0.0
        else:
            body_top = int(h * 0.35)
            final_bg_mask[body_top:h, :] = 0.0

        fg_alpha = ((1.0 - final_bg_mask) * 255).astype(np.uint8)
        arr_rgba = np.dstack((arr, fg_alpha))
        return Image.fromarray(arr_rgba, mode="RGBA")
    except Exception:
        return pil_img.convert("RGBA")


def replace_background_color(pil_img, hex_color):
    """
    AnyEraser-style AI Deep Learning Background Removal & Color Replacement.
    Isolates the subject/person completely using U2-Net deep learning portrait segmentation,
    removes 100% of the entire original background behind the person, and composites
    the person onto a new solid canvas of the target hex_color.
    Face, skin, hair, shirt, suit, and tie are 100% protected and NEVER recolored.
    - hex_color: string (e.g. '#ffffff', '#3b82f6', '#ef4444')
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

    cutout_rgba = get_subject_cutout(pil_img)
    bg_canvas = Image.new("RGBA", cutout_rgba.size, target_rgb + (255,))
    composite = Image.alpha_composite(bg_canvas, cutout_rgba)
    return composite.convert("RGB")


def process_photo_edits(image_bytes, brightness=0.0, width=None, height=None, sharpness=1.0, bg_color=None):
    """
    Applies brightness, resolution resize, sharpness, and background color adjustments to an image.
    - brightness: float offset (-100 to +100) or factor (e.g. 1.0 = default)
    - width, height: target dimensions in pixels (optional)
    - sharpness: factor (1.0 = default, > 1.0 sharpens, e.g. 1.5 - 3.0)
    - bg_color: hex string (e.g. '#ffffff', '#3b82f6')
    Returns: JPEG image bytes
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    try:
        from PIL import ImageOps
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    pil_img = pil_img.convert("RGB")

    # 1. Background Color Replacement
    if bg_color:
        pil_img = replace_background_color(pil_img, bg_color)

    # 2. Resolution / Resizing
    if width and height:
        try:
            w = int(width)
            h = int(height)
            if w > 0 and h > 0:
                pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
        except (ValueError, TypeError):
            pass

    # 3. Brightness adjustment
    try:
        b_val = float(brightness)
        if b_val != 0.0:
            if -100 <= b_val <= 100 and b_val != 1.0:
                factor = max(0.1, 1.0 + (b_val / 100.0))
            else:
                factor = max(0.1, b_val)
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(factor)
    except (ValueError, TypeError):
        pass

    # 4. Sharpness adjustment
    try:
        s_val = float(sharpness)
        if s_val != 1.0 and s_val >= 0:
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(s_val)
    except (ValueError, TypeError):
        pass

    out_buf = io.BytesIO()
    pil_img.save(out_buf, format="JPEG", quality=95)
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
