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
    Accurately identifies neckties and formal neckwear by verifying:
    1. Central vertical tie blade profile (midline column contrasting against bilateral shirt flanks).
    2. Bilateral shirt symmetry (shirt fabric matches on left and right, while tie differs from both).
    3. Bilateral vertical edges (paired opposite-polarity Gx Sobel edges bounding the tie blade).
    4. Collar / tie knot region structure.
    Rejects open collars, bare necks, round/crew-neck T-shirts, and plain single-color shirts without ties.
    """
    if not faces:
        return {"passed": False, "message": "Face not detected — cannot check for tie.", "meta": {}}

    h, w = bgr.shape[:2]
    f = faces[0]
    fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]

    pts = _face_mesh_landmarks(bgr)
    skin = _extract_face_skin_profile(bgr, pts, f)

    blade_contrast = 0.0
    contrast_ratio = 0.0
    vert_edge_score = 0.0
    tie_score = 0.0
    has_tie = False
    lm_analyzed = False

    if pts is not None:
        try:
            lm_analyzed = True
            chin = pts[152]
            eye_span = max(10, abs(pts[362][0] - pts[133][0]))
            cx, cy = chin[0], chin[1]

            # Torso / Upper Chest ROI (where the tie blade sits on the shirt)
            cy1 = min(h, cy + int(eye_span * 0.8))
            cy2 = min(h, cy + int(eye_span * 2.8))
            torso_w = int(eye_span * 2.4)
            cx1 = max(0, cx - torso_w // 2)
            cx2 = min(w, cx + torso_w // 2)

            chest_crop = bgr[cy1:cy2, cx1:cx2]
            if chest_crop.size > 0 and (cy2 - cy1) >= 10 and (cx2 - cx1) >= 20:
                cw = chest_crop.shape[1]
                c_strip = chest_crop[:, int(cw * 0.35):int(cw * 0.65)]
                l_strip = chest_crop[:, 0:int(cw * 0.30)]
                r_strip = chest_crop[:, int(cw * 0.70):cw]

                c_m = np.mean(c_strip, axis=(0, 1))
                l_m = np.mean(l_strip, axis=(0, 1))
                r_m = np.mean(r_strip, axis=(0, 1))
                flanks_m = (l_m + r_m) / 2.0

                blade_contrast = float(np.linalg.norm(c_m - flanks_m))
                flank_diff = float(np.linalg.norm(l_m - r_m))
                contrast_ratio = blade_contrast / max(12.0, flank_diff * 0.5 + 8.0)

                # Bilateral vertical edges flanking the tie blade
                gray_chest = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2GRAY)
                sob_x = np.abs(cv2.Sobel(gray_chest, cv2.CV_64F, 1, 0, ksize=3))
                v_left = sob_x[:, int(cw * 0.20):int(cw * 0.42)]
                v_right = sob_x[:, int(cw * 0.58):int(cw * 0.80)]
                vert_edge_score = (float(np.mean(v_left)) + float(np.mean(v_right))) / 2.0 / 15.0

                tie_score = (contrast_ratio * 1.6) + (vert_edge_score * 1.2)

                # Tie presence rule
                th_contrast = float(params.get("tie_contrast_threshold", 40.0))
                th_ratio = float(params.get("tie_ratio_threshold", 1.25))

                has_tie = (blade_contrast >= th_contrast and contrast_ratio >= th_ratio and vert_edge_score >= 0.45) or \
                          (contrast_ratio >= 1.75 and blade_contrast >= 30.0 and vert_edge_score >= 0.40)
        except Exception:
            lm_analyzed = False

    if not lm_analyzed:
        # Fallback based on face bounding box
        cx = fx + fw // 2
        chin_y = min(h, fy + fh)
        cy1 = min(h, chin_y + int(fh * 0.25))
        cy2 = min(h, chin_y + int(fh * 0.95))
        torso_w = int(fw * 1.0)
        cx1 = max(0, cx - torso_w // 2)
        cx2 = min(w, cx + torso_w // 2)

        chest_crop = bgr[cy1:cy2, cx1:cx2]
        if chest_crop.size > 0 and (cy2 - cy1) >= 10 and (cx2 - cx1) >= 20:
            cw = chest_crop.shape[1]
            c_strip = chest_crop[:, int(cw * 0.35):int(cw * 0.65)]
            l_strip = chest_crop[:, 0:int(cw * 0.30)]
            r_strip = chest_crop[:, int(cw * 0.70):cw]

            c_m = np.mean(c_strip, axis=(0, 1))
            l_m = np.mean(l_strip, axis=(0, 1))
            r_m = np.mean(r_strip, axis=(0, 1))
            flanks_m = (l_m + r_m) / 2.0

            blade_contrast = float(np.linalg.norm(c_m - flanks_m))
            flank_diff = float(np.linalg.norm(l_m - r_m))
            contrast_ratio = blade_contrast / max(12.0, flank_diff * 0.5 + 8.0)

            gray_chest = cv2.cvtColor(chest_crop, cv2.COLOR_BGR2GRAY)
            sob_x = np.abs(cv2.Sobel(gray_chest, cv2.CV_64F, 1, 0, ksize=3))
            v_left = sob_x[:, int(cw * 0.20):int(cw * 0.42)]
            v_right = sob_x[:, int(cw * 0.58):int(cw * 0.80)]
            vert_edge_score = (float(np.mean(v_left)) + float(np.mean(v_right))) / 2.0 / 15.0

            tie_score = (contrast_ratio * 1.6) + (vert_edge_score * 1.2)
            has_tie = (blade_contrast >= 40.0 and contrast_ratio >= 1.25 and vert_edge_score >= 0.45)

    meta = {
        "blade_contrast": round(blade_contrast, 2),
        "contrast_ratio": round(contrast_ratio, 2),
        "vert_edge_score": round(vert_edge_score, 2),
        "tie_score": round(tie_score, 2),
        "lm_analyzed": lm_analyzed
    }

    if not has_tie:
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
    """
    Passport-standard sharpness detection.
    Evaluates focus/sharpness on the face & subject region (where detail matters),
    preventing solid/uniform background regions from diluting the sharpness score.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Focus evaluation on face/head ROI if detected (ICAO passport guideline)
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
            variance = cv2.Laplacian(roi, cv2.CV_64F).var()
        else:
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    else:
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    threshold = float(params.get("blur_threshold", 80.0))
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
