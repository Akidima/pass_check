# University Passport Photo Verification System

A production-style passport/ID photo checker for university enrollment systems.
Students upload a photo **or** capture one live via webcam; the system runs it
through an AI computer-vision pipeline and instantly reports pass/fail against
criteria the admin controls.

## Architecture

```
pass-check/
  php-app/            PHP + CSS + JS web application (student portal + admin panel)
    index.php          Student portal (upload / camera capture / results)
    admin/              Admin login, dashboard (toggle criteria), submissions log, settings
    api/verify.php      Bridges PHP <-> Python verification service, stores results
    includes/           DB (SQLite via PDO), auth, shared header
    assets/             CSS + JS
    data/                SQLite database file (auto-created)
    uploads/            Stored submitted photos

  python-service/      Python Flask microservice — the actual computer-vision engine
    app.py              HTTP API (/verify, /health)
    verify.py           All CV checks (OpenCV + MediaPipe)
    requirements.txt
```

**Why two languages?** PHP handles the web app, auth, database, and admin UI —
but detecting glasses, counting faces, judging background whiteness, tie
presence, head pose, blur, etc. requires real computer vision. PHP has no
library for that, so a Python microservice (OpenCV + MediaPipe) does the
image analysis and returns structured JSON that PHP consumes and stores.

## Verification Criteria (toggle any on/off from the Admin dashboard)

| Criteria | What it checks |
|---|---|
| Single Face Detection | Exactly one person in frame |
| Face Size & Centering | Face properly framed, not too close/far/off-center |
| No Eyeglasses | Flags detected eyewear |
| Strictly White Background | Background brightness + uniformity |
| Tie / Formal Neckwear Required | Heuristic neckwear detection below chin |
| Minimum Resolution | Width/height thresholds |
| Passport Size Aspect Ratio | Matches standard 35:45mm-style ratio |
| Sharpness (No Blur) | Laplacian variance sharpness check |
| Proper Lighting / Exposure | Rejects too dark / overexposed photos |
| Straight Head Pose | Flags tilted/turned heads via face mesh |
| Eyes Open | Eye-aspect-ratio check |

## Strict white-background verification

The white-background check is a deterministic OpenCV pipeline. The repository
does not contain a labelled background dataset, so it does not claim to use a
trained model or a fabricated confidence metric.

For each image, it samples the complete outer border while excluding a
conservative region around any detected face and shoulders. A sampled pixel
counts as acceptable white when it qualifies through either of two
administrator-controlled tiers:

- **Pure white (always active)** — RGB brightness, HSV saturation, and
  CIE-LAB distance to pure white within the configured limits. This is the
  classic studio rule and still gates every strictness level.
- **Near white (opt-in per level)** — slightly darker or lightly tinted by
  warm/cool lighting but objectively near-white in CIE-LAB: high lightness
  (L*), low chroma C*, and the tint confined to the neutral warm/cool (b*)
  axis. This accepts light-grey studio walls, white walls under household
  lighting, and exposure/compression drift. It uses objective image
  characteristics only — no nationality-, region-, or demographic-based
  rules — and never admits beige, cream, blue, green, or dark surfaces.

Independent hard guards run on every sampled pixel regardless of tier and can
never be relaxed into accepting an unsuitable background: dark/black coverage,
chromatic coverage, largest contiguous non-white component, total white
coverage, and background luminance uniformity (p90–p10 of L*, which tolerates
ordinary window-light gradients while still catching real shadows). A
compliance failure for an enabled **Strictly White Background** criterion
always blocks overall approval, even if the separate general pass-count
threshold has been met.

Admin → Settings controls background validation through EXACTLY TWO settings:

| Control | Values | Effect |
|---|---|---|
| **White Background Strictness Level** | `strict` · `standard` · `relaxed` · `accept_all` | Selects one preset that supplies every image-analysis threshold (brightness / saturation / ΔE limits, the near-white band, coverage gates, and the dark/coloured guards) |
| **Near-White Background Acceptance** | ON / OFF toggle | ON admits bright, almost colourless backgrounds photographed under real-world lighting (slightly dimmed or warm/cool-tinted white walls, light-grey studio backdrops). OFF enforces pure-white-only criteria |

Level behaviour (`BACKGROUND_STRICTNESS_LEVELS` in
`python-service/verify.py` is the single authoritative definition):

| Level | Tier-1 limits | Near-white band | Coverage gates | Intended use |
|---|---|---|---|---|
| `strict` | value ≥ 235, sat ≤ 18, ΔE ≤ 10 | disabled by default | ≥ 30% white, ≤ 30% patch | International passport / NYSC: pure studio white only |
| `standard` (default) | value ≥ 215, sat ≤ 28, ΔE ≤ 16 | L* ≥ 90, C* ≤ 13, \|b\*\| ≤ 13 | ≥ 60% white, ≤ 30% patch | Admission portals: light-grey walls, warm/cool lighting, mild shadows OK |
| `relaxed` | value ≥ 180, sat ≤ 45, ΔE ≤ 25 | L* ≥ 86, C* ≤ 15, \|b\*\| ≤ 14 | ≥ 15% white, ≤ 60% patch | Home-taken photos with imperfect lighting |
| `accept_all` | wide | wide | only the dark/coloured guards matter | Only rejects dark or vividly coloured backgrounds |

The **Near-White Background Acceptance** toggle overrides the level default:
ON forces the tier on at every level (Strict included), OFF forces it off
everywhere (legacy pure-white-only behaviour). Beige, cream, blue, green and
dark backgrounds are rejected regardless of both settings.

The former per-field "Advanced Threshold" overrides were removed: they formed
a second configuration source whose persisted values silently beat the chosen
level, which is why changing the two settings above previously had no effect.
The PHP API now forwards only the two controls; the Python service ignores
and strips any per-field `bg_*` keys at its `/verify` boundary, and old
databases are migrated (legacy rows deleted, an explicit legacy near-white
choice carried over) on startup.

Each result stores the sampled-pixel count, white/non-white coverage (split
into pure-white and near-white coverage), largest contamination component,
LAB distance, luminance range, deterministic quality score, and thresholds
used in `results.white_background.meta`.

Run the focused regression tests with:

```bash
python-service/venv/bin/python -m unittest discover -s python-service/tests -v
php tests/test_background_settings.php   # admin-settings persistence + migration
```

For a reproducible local CPU-only timing run (without face detection or HTTP
upload time), use:

```bash
python-service/venv/bin/python python-service/benchmarks/benchmark_white_background.py
```

The `/verify` JSON response also includes `timings_ms` for image decode, face
detection, every enabled check, and total service-side processing. The optional
U2-Net/rembg model is initialized lazily and cached only for `/edit-photo`, so
it does not delay photo validation.

## Setup & Run (Local / XAMPP-style)

### 1. Python verification service

```bash
cd python-service
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python app.py
```

Runs on `http://127.0.0.1:5001`. Verify with `http://127.0.0.1:5001/health`.

### Tie detection deployment contract

The repository intentionally ships without a university-specific tie model.
By default (`TIE_DETECTOR_BACKEND=auto`), it uses torchvision's maintained
COCO detector, which has a built-in **traditional necktie** class, so the
application works immediately. A visible chest ROI with a valid conventional
tie box passes; an ROI without one fails the required-tie criterion. The
face-relative localization gate rejects high-scoring tie-like objects on a
lapel, background, or another garment. Set `TIE_DETECTOR_BACKEND=custom` only
after deploying the calibrated custom model described below.

Deploy a model only with its adjacent policy file:

```text
python-service/models/tie_detector_v1.pt
python-service/models/tie_detector_v1.policy.json
```

Start from [`python-service/models/tie_detector.policy.example.json`](python-service/models/tie_detector.policy.example.json).
The policy binds the custom model SHA-256, calibrated positive threshold,
face-relative box limits, and held-out metrics. The service refuses a missing,
stale, malformed, or below-target policy. Set `TIE_MODEL_PATH` and, when the
policy is not adjacent to the model, `TIE_MODEL_POLICY_PATH`.

The training set must contain both tie and no-tie passport photos, identity
disjoint train/validation/test splits, and hard negatives (open collars,
V-necks, scarves, necklaces, lanyards, patterned shirts, lapels, and visible
background objects). Label full tie boxes consistently, including the knot.
Training images must use the same `face_relative_upper_body_v1` crop emitted
by `UpperBodyVisibilityEstimator` at inference; do not train on full portraits
and infer on chest crops. Transform the annotation coordinates when producing
those crops, discard boxes outside a crop, and retain no-tie crops as empty
annotations. Calibrate the score and geometry only on validation data, then report the
held-out test metrics before promotion. `training/evaluate_tie_detector.py`
now counts a tie-image prediction as correct only when it overlaps the labeled
tie at the configured IoU; a box elsewhere can no longer inflate recall.

> First run of MediaPipe/OpenCV installs may take a few minutes. Requires Python 3.9–3.12 (mediapipe wheels).

### 2. PHP web app

```bash
cd php-app
php -S localhost:8000
```

Then open:
- Student portal: http://localhost:8000/
- Admin panel: http://localhost:8000/admin/login.php
  - Default login: `admin` / `Admin@123` (change immediately in Settings)

The SQLite database (`data/database.sqlite`) and criteria/settings are
auto-created on first request — no manual DB setup needed.

### 3. Point PHP at the Python service (if not default)

Admin → Settings → "Python Verification Service URL" (default `http://127.0.0.1:5001`).

## Deploying to a real university server

- Run the Python service as a persistent process (e.g. via `systemd`, `pm2`,
  `gunicorn` + `nginx` reverse proxy, or Windows Task Scheduler / NSSM).
- Serve the PHP app via Apache/Nginx + PHP-FPM (XAMPP works for a pilot).
- Switch SQLite → MySQL by adjusting `includes/db.php` if concurrent load requires it.
- Put both behind HTTPS; only expose the PHP app publicly — keep the Python
  service on localhost/internal network, reachable only by the PHP server.
- Camera capture requires HTTPS (or localhost) per browser security policy.

## Security notes

- Admin passwords are hashed with `password_hash()` (bcrypt).
- File uploads are MIME-validated server-side (not just by extension).
- The criteria enforced are always read from the server-side database — the
  client cannot spoof which checks apply.
- Change the default admin password before any real deployment.
