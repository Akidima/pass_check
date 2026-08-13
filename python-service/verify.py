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
import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    MEDIAPIPE_AVAILABLE = False

# Fallback Haar cascade for face detection when mediapipe isn't installed
_HAAR_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- Helpers ----------

def _load_image(image_bytes):
    """Load image bytes into an OpenCV BGR array, correcting EXIF orientation."""
    pil_img = Image.open(io.BytesIO(image_bytes))
    try:
        from PIL import ImageOps
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


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


def _detect_faces(bgr):
    """Return list of face detections with bounding boxes (relative + absolute).
    Uses mediapipe when available, otherwise falls back to OpenCV's Haar cascade.
    Applies NMS to filter out duplicate overlapping boxes and tiny false positives."""
    h, w = bgr.shape[:2]

    faces = []
    if MEDIAPIPE_AVAILABLE:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as fd:
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
    else:
        # Haar cascade fallback
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections = _HAAR_FACE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        for (x, y, fw, fh) in detections:
            faces.append({"x": int(x), "y": int(y), "w": int(fw), "h": int(fh), "score": 1.0, "keypoints": None})

    return _nms_faces(faces, h, w)


def _face_mesh_landmarks(bgr):
    """Fine-grained facial landmarks via mediapipe FaceMesh. Returns None if
    mediapipe is unavailable — callers must handle that gracefully (checks
    that depend on it will skip rather than fail)."""
    if not MEDIAPIPE_AVAILABLE:
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                refine_landmarks=True, min_detection_confidence=0.5) as fm:
        results = fm.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        return pts


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
    Uses multi-signal corroboration:
    1. MediaPipe facial landmark nose bridge & eye socket geometry (if available).
    2. Sobel horizontal gradient ($G_y$) & Canny edge density across nose bridge.
    3. Multi-scale Haar cascade detection with consensus requirements.
    Requires multi-signal consensus before flagging eyeglasses to eliminate false positives on bare eyes.
    """
    if not faces:
        return {"passed": False, "message": "Face not detected — cannot check for glasses.", "meta": {}}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = faces[0]
    fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]
    h, w = bgr.shape[:2]

    pts = _face_mesh_landmarks(bgr)

    bridge_canny = 0.0
    bridge_sobel = 0.0
    bridge_std = 0.0
    eye_canny = 0.0
    eye_sobel = 0.0
    lm_analyzed = False

    if pts is not None:
        try:
            lm_analyzed = True
            p_l_inner = pts[133]
            p_r_inner = pts[362]
            p_l_outer = pts[33]
            p_r_outer = pts[263]

            # Bridge ROI between inner eye corners
            bx1 = max(0, min(p_l_inner[0], p_r_inner[0]))
            bx2 = min(w, max(p_l_inner[0], p_r_inner[0]))
            by1 = max(0, min(p_l_inner[1], p_r_inner[1]) - int(fh * 0.04))
            by2 = min(h, max(p_l_inner[1], p_r_inner[1]) + int(fh * 0.06))

            bridge_roi = gray[by1:by2, bx1:bx2]
            if bridge_roi.size > 0:
                edges_br = cv2.Canny(bridge_roi, 50, 130)
                bridge_canny = float(np.mean(edges_br > 0))
                sobel_y = cv2.Sobel(bridge_roi, cv2.CV_64F, 0, 1, ksize=3)
                bridge_sobel = float(np.mean(np.abs(sobel_y)))
                bridge_std = float(np.std(bridge_roi))

            ex1 = max(0, p_l_outer[0] - int(fw * 0.05))
            ex2 = min(w, p_r_outer[0] + int(fw * 0.05))
            ey1 = max(0, min(p_l_outer[1], p_r_outer[1]) - int(fh * 0.08))
            ey2 = min(h, max(p_l_outer[1], p_r_outer[1]) + int(fh * 0.12))

            eye_roi = gray[ey1:ey2, ex1:ex2]
            if eye_roi.size > 0:
                edges_eye = cv2.Canny(eye_roi, 50, 130)
                eye_canny = float(np.mean(edges_eye > 0))
                sobel_eye_y = cv2.Sobel(eye_roi, cv2.CV_64F, 0, 1, ksize=3)
                eye_sobel = float(np.mean(np.abs(sobel_eye_y)))
        except Exception:
            lm_analyzed = False

    if not lm_analyzed or bridge_canny == 0.0:
        bx1 = max(0, fx + int(fw * 0.38))
        bx2 = min(w, fx + int(fw * 0.62))
        by1 = max(0, fy + int(fh * 0.28))
        by2 = min(h, fy + int(fh * 0.44))

        bridge_roi = gray[by1:by2, bx1:bx2]
        if bridge_roi.size > 0:
            edges_br = cv2.Canny(bridge_roi, 50, 130)
            bridge_canny = float(np.mean(edges_br > 0))
            sobel_y = cv2.Sobel(bridge_roi, cv2.CV_64F, 0, 1, ksize=3)
            bridge_sobel = float(np.mean(np.abs(sobel_y)))
            bridge_std = float(np.std(bridge_roi))

        ey1 = max(0, fy + int(fh * 0.22))
        ey2 = min(h, fy + int(fh * 0.50))
        ex1 = max(0, fx + int(fw * 0.10))
        ex2 = min(w, fx + int(fw * 0.90))

        eye_roi = gray[ey1:ey2, ex1:ex2]
        if eye_roi.size > 0:
            edges_eye = cv2.Canny(eye_roi, 50, 130)
            eye_canny = float(np.mean(edges_eye > 0))
            sobel_eye_y = cv2.Sobel(eye_roi, cv2.CV_64F, 0, 1, ksize=3)
            eye_sobel = float(np.mean(np.abs(sobel_eye_y)))

    # Multi-scale Haar Cascade for eyeglasses (with high minNeighbors=5 to reduce false hits)
    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    upper_face = gray[max(0, fy):min(h, fy + int(fh * 0.65)), max(0, fx):min(w, fx + fw)]
    eyes_tree = []
    if upper_face.size > 0:
        eyes_tree = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    # Tuned thresholds for actual eyeglasses frames
    th_bridge_canny = float(params.get("glasses_bridge_canny", 0.085))
    th_bridge_sobel = float(params.get("glasses_bridge_sobel", 20.0))
    th_bridge_std = float(params.get("glasses_bridge_std", 32.0))
    th_eye_canny = float(params.get("glasses_eye_canny", 0.14))
    th_eye_sobel = float(params.get("glasses_eye_sobel", 24.0))

    bridge_detected = (bridge_canny > th_bridge_canny) and (bridge_sobel > th_bridge_sobel or bridge_std > th_bridge_std)
    eye_edges_detected = (eye_canny > th_eye_canny) and (eye_sobel > th_eye_sobel)
    cascade_detected = len(eyes_tree) >= 2

    # Require corroborating signals (multi-signal consensus)
    glasses_detected = (bridge_detected and eye_edges_detected) or \
                       (cascade_detected and (bridge_detected or eye_edges_detected)) or \
                       (bridge_canny > 0.14 and bridge_sobel > 28.0)

    meta = {
        "bridge_canny": round(bridge_canny, 4),
        "bridge_sobel": round(bridge_sobel, 2),
        "bridge_std": round(bridge_std, 2),
        "eye_canny": round(eye_canny, 4),
        "eye_sobel": round(eye_sobel, 2),
        "cascade_eyes": len(eyes_tree),
        "lm_analyzed": lm_analyzed
    }

    if glasses_detected:
        return {"passed": False, "message": "Eyeglasses detected. Please remove glasses for your passport photo.", "meta": meta}
    return {"passed": True, "message": "No eyeglasses detected.", "meta": meta}


def check_white_background(bgr, faces, params):
    """
    Production-grade white-background verification.
    Uses corner-focused background sampling, head/hair & shoulder exclusion masks,
    and foreground pixel filtering to accurately verify white/off-white background.
    """
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim != 3:
        return {"passed": False, "message": "Invalid image data.", "meta": {}}

    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return {"passed": False, "message": "Invalid image dimensions.", "meta": {}}

    # 1. Build corner-focused background sampling mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Top-left and top-right background corners
    corner_h = max(1, int(h * 0.28))
    corner_w = max(1, int(w * 0.24))

    mask[0:corner_h, 0:corner_w] = 255
    mask[0:corner_h, w - corner_w:w] = 255
    # Thin top border strip outside middle head zone
    mask[0:max(1, int(h * 0.08)), :] = 255

    # 2. Exclude head, hair, and shoulder regions
    if faces:
        try:
            face = max(faces, key=lambda f: float(f.get("w", 0)) * float(f.get("h", 0)))
            fx = int(face.get("x", 0))
            fy = int(face.get("y", 0))
            fw = int(face.get("w", 0))
            fh = int(face.get("h", 0))

            if fw > 0 and fh > 0:
                # Exclude head top, face, neck, and shoulders
                ex_x1 = max(0, fx - int(fw * 0.55))
                ex_x2 = min(w, fx + fw + int(fw * 0.55))
                ex_y1 = 0  # Zero out region above head to prevent hair sampling
                ex_y2 = min(h, fy + fh + int(fh * 1.0))

                mask[ex_y1:ex_y2, ex_x1:ex_x2] = 0
        except Exception:
            pass

    sampled_pixel_count = int(np.count_nonzero(mask))
    if sampled_pixel_count < 60:
        # Fall back to outer margin sampling if face occupies most of frame
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[0:max(1, int(h * 0.12)), 0:max(1, int(w * 0.15))] = 255
        mask[0:max(1, int(h * 0.12)), w - max(1, int(w * 0.15)):w] = 255
        sampled_pixel_count = int(np.count_nonzero(mask))
        if sampled_pixel_count < 10:
            return {"passed": True, "message": "Background region occupied by face framing.", "meta": {}}

    # 3. Color space conversions
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    lab_l = lab[:, :, 0].astype(np.float32)

    sampled_saturation = saturation[mask > 0]
    sampled_value = value[mask > 0]
    sampled_luminance = lab_l[mask > 0]

    # Filter out dark foreground/hair edge pixels (Value < 75) to evaluate pure background pixels
    valid_bg = sampled_value >= 75.0
    if np.count_nonzero(valid_bg) >= 20:
        eval_sat = sampled_saturation[valid_bg]
        eval_val = sampled_value[valid_bg]
        eval_lum = sampled_luminance[valid_bg]
    else:
        eval_sat = sampled_saturation
        eval_val = sampled_value
        eval_lum = sampled_luminance

    # 4. Thresholds & Classification
    min_value = int(params.get("bg_min_value", 135))
    max_saturation = int(params.get("bg_max_saturation", 65))
    min_white_coverage = float(params.get("bg_min_white_coverage", 0.60))
    max_colored_coverage = float(params.get("bg_max_colored_coverage", 0.15))
    max_lum_std = float(params.get("bg_max_lum_std", 40.0))

    white_pixels = (eval_val >= min_value) & (eval_sat <= max_saturation)
    white_coverage = float(np.mean(white_pixels))

    # Strongly colored background pixels (e.g., solid blue, red, green wall)
    colored_pixels = (eval_sat > int(params.get("bg_colored_sat", 80))) & (eval_val > 70)
    colored_coverage = float(np.mean(colored_pixels))

    mean_value = float(np.mean(eval_val))
    luminance_std = float(np.std(eval_lum))

    issues = []
    if colored_coverage > max_colored_coverage:
        issues.append("Colored background detected. Background must be plain white.")
    elif mean_value < min_value:
        issues.append("Background is too dark. Please use a bright white background.")
    elif white_coverage < min_white_coverage:
        issues.append("Background is not sufficiently white.")

    if luminance_std > max_lum_std:
        issues.append("Background is not sufficiently uniform.")

    passed = len(issues) == 0

    meta = {
        "sampled_pixels": sampled_pixel_count,
        "white_coverage": round(white_coverage, 3),
        "white_coverage_percent": round(white_coverage * 100.0, 2),
        "colored_coverage": round(colored_coverage, 3),
        "colored_coverage_percent": round(colored_coverage * 100.0, 2),
        "mean_value": round(mean_value, 2),
        "luminance_std": round(luminance_std, 2)
    }

    if passed:
        return {"passed": True, "message": "Background is plain white and uniform.", "meta": meta}

    return {"passed": False, "message": " ".join(issues), "meta": meta}


def check_tie(bgr, faces, params):
    """
    Structural Tie & Formal Neckwear Detector.
    Determines tie presence by checking:
    1. Skin presence directly below chin (open/bare neck vs tie knot).
    2. Central narrow vertical band contrasting against left/right shirt sides.
    3. Bilateral vertical edges flanking the center of the chest.
    """
    if not faces:
        return {"passed": False, "message": "Face not detected — cannot check for tie.", "meta": {}}

    h, w = bgr.shape[:2]
    f = faces[0]
    fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]
    cx = fx + fw / 2.0
    chin_y = min(h, fy + fh)

    region_y1 = int(chin_y + fh * 0.05)
    region_y2 = min(h, int(chin_y + fh * 0.95))
    region_w = int(fw * 0.9)
    region_x1 = max(0, int(cx - region_w / 2.0))
    region_x2 = min(w, int(cx + region_w / 2.0))

    if region_y2 <= region_y1 or region_x2 <= region_x1:
        return {"passed": False, "message": "Cannot evaluate neckwear region.", "meta": {}}

    neck_region = bgr[region_y1:region_y2, region_x1:region_x2]
    if neck_region.size == 0:
        return {"passed": False, "message": "Cannot evaluate neckwear region.", "meta": {}}

    nh, nw = neck_region.shape[:2]

    # 1. Skin tone detection directly below chin (upper neck area)
    upper_neck = neck_region[0:max(1, int(nh * 0.4)), max(0, int(nw * 0.25)):min(nw, int(nw * 0.75))]
    skin_ratio = 0.0
    if upper_neck.size > 0:
        hsv_un = cv2.cvtColor(upper_neck, cv2.COLOR_BGR2HSV)
        h_channel = hsv_un[:, :, 0]
        s_channel = hsv_un[:, :, 1]
        v_channel = hsv_un[:, :, 2]
        # Skin HSV mask
        skin_mask = (((h_channel <= 22) | (h_channel >= 165)) & 
                     (s_channel >= 25) & (s_channel <= 170) & 
                     (v_channel >= 60))
        skin_ratio = float(np.mean(skin_mask))

    # 2. Central narrow vertical band vs Left/Right shirt sides analysis
    c_x1 = max(0, int(nw * 0.35))
    c_x2 = min(nw, int(nw * 0.65))
    left_x2 = max(1, int(nw * 0.30))
    right_x1 = min(nw - 1, int(nw * 0.70))

    center_strip = neck_region[:, c_x1:c_x2]
    left_strip = neck_region[:, 0:left_x2]
    right_strip = neck_region[:, right_x1:nw]

    contrast_score = 0.0
    if center_strip.size > 0 and left_strip.size > 0 and right_strip.size > 0:
        gray_neck = cv2.cvtColor(neck_region, cv2.COLOR_BGR2GRAY)
        center_val = float(np.mean(gray_neck[:, c_x1:c_x2]))
        sides_val = (float(np.mean(gray_neck[:, 0:left_x2])) + float(np.mean(gray_neck[:, right_x1:nw]))) / 2.0
        val_diff = abs(center_val - sides_val)

        center_bgr = np.mean(center_strip, axis=(0, 1))
        sides_bgr = (np.mean(left_strip, axis=(0, 1)) + np.mean(right_strip, axis=(0, 1))) / 2.0
        color_diff = float(np.linalg.norm(center_bgr - sides_bgr))

        contrast_score = (val_diff / 50.0) + (color_diff / 60.0)

    # 3. Vertical edges (Sobel dx=1) flanking center strip
    gray_neck = cv2.cvtColor(neck_region, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_neck, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = np.abs(sobel_x)

    edge_zone_left = abs_sobel_x[:, max(0, int(nw * 0.20)):min(nw, int(nw * 0.40))]
    edge_zone_right = abs_sobel_x[:, max(0, int(nw * 0.60)):min(nw, int(nw * 0.80))]
    vert_edge_score = 0.0
    if edge_zone_left.size > 0 and edge_zone_right.size > 0:
        vert_edge_score = (float(np.mean(edge_zone_left)) + float(np.mean(edge_zone_right))) / 2.0 / 25.0

    raw_tie_score = contrast_score + vert_edge_score
    tie_score = raw_tie_score - (skin_ratio * 1.5)
    threshold = float(params.get("tie_score_threshold", 0.65))

    # Tie is detected IF:
    # 1. Bare skin does not dominate upper neck (skin_ratio < 0.50) AND tie_score >= threshold, OR
    # 2. Structural tie features are very strong (raw_tie_score >= 1.4 and contrast_score >= 0.8)
    tie_detected = (skin_ratio < 0.50 and tie_score >= threshold) or (raw_tie_score >= 1.4 and contrast_score >= 0.8)

    meta = {
        "tie_score": round(tie_score, 3),
        "skin_ratio": round(skin_ratio, 3),
        "contrast_score": round(contrast_score, 3),
        "vert_edge_score": round(vert_edge_score, 3)
    }

    if not tie_detected:
        return {"passed": False, "message": "Tie not detected. This institution requires formal attire with a tie.", "meta": meta}
    return {"passed": True, "message": "Tie / formal neckwear detected.", "meta": meta}


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
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    threshold = params.get("blur_threshold", 80.0)
    if variance < threshold:
        return {"passed": False, "message": "Photo appears blurry. Please retake with a steady camera and good focus.",
                "meta": {"sharpness": float(variance)}}
    return {"passed": True, "message": "Photo sharpness is acceptable.", "meta": {"sharpness": float(variance)}}


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
        return {"passed": True, "message": "Head pose check skipped (landmarks unavailable).", "meta": {}}

    # Landmark indices: left eye outer (33), right eye outer (263), nose tip (1), chin (152)
    try:
        left_eye = np.array(pts[33])
        right_eye = np.array(pts[263])
        nose = np.array(pts[1])
        chin = np.array(pts[152])
    except IndexError:
        return {"passed": True, "message": "Head pose check skipped.", "meta": {}}

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
        return {"passed": True, "message": "Eyes-open check skipped (landmarks unavailable).", "meta": {}}
    try:
        # Left eye vertical landmarks: 159 (top), 145 (bottom); horizontal: 33,133
        top = np.array(pts[159]); bottom = np.array(pts[145])
        left = np.array(pts[33]); right = np.array(pts[133])
        vert = np.linalg.norm(top - bottom)
        horiz = np.linalg.norm(left - right) + 1e-6
        ear = vert / horiz
    except IndexError:
        return {"passed": True, "message": "Eyes-open check skipped.", "meta": {}}

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
    "min_resolution": "Minimum Resolution",
    "passport_ratio": "Passport Size Aspect Ratio",
    "no_blur": "Sharpness (No Blur)",
    "brightness": "Proper Lighting / Exposure",
    "head_pose": "Straight Head Pose",
    "eyes_open": "Eyes Open",
}


def run_checks(image_bytes, enabled_criteria, params=None):
    """
    enabled_criteria: dict {criteria_key: bool}
    params: dict of tunable numeric parameters (optional, merges with defaults)
    """
    params = params or {}
    try:
        bgr = _load_image(image_bytes)
    except Exception as e:
        return {
            "overall_passed": False,
            "error": f"Could not read image: {str(e)}",
            "results": {}
        }

    faces = _detect_faces(bgr)

    results = {}
    overall_passed = True

    # Always run single_face first if enabled, others depend on face presence but degrade gracefully
    ordered_keys = [k for k in CHECK_REGISTRY if enabled_criteria.get(k)]

    passed_count = 0
    for key in ordered_keys:
        fn = CHECK_REGISTRY[key]
        try:
            res = fn(bgr, faces, params)
        except Exception as e:
            res = {"passed": False, "message": f"Check '{key}' failed to run: {str(e)}", "meta": {}}
        res["label"] = CHECK_LABELS.get(key, key)
        results[key] = res
        if res.get("passed"):
            passed_count += 1

    total_criteria = len(ordered_keys)
    min_required_param = int(params.get("min_pass_criteria", 4))
    if total_criteria == 0:
        overall_passed = True
        required_to_pass = 0
    else:
        required_to_pass = min(min_required_param, total_criteria)
        overall_passed = (passed_count >= required_to_pass)

    return {
        "overall_passed": overall_passed,
        "passed_count": passed_count,
        "total_criteria": total_criteria,
        "required_to_pass": required_to_pass,
        "results": results,
        "image_info": {
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
            "faces_detected": len(faces)
        }
    }
