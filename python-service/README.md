# University Photo Verification Service - Technical Audit & Recommendation

**Audit date:** 2026-08-28

**Scope:** Repository inspection and production-deployment research only. This
document describes the current worktree and proposed changes. It is not an
approval to implement additional recommendations.

## 1. Executive Summary

This repository contains a PHP university portal and a Python computer-vision
service. The Python service is a synchronous, inference-only Flask application
designed to receive an image from PHP, run several explainable checks, and
return a JSON verdict.

The current Python pipeline combines:

- MediaPipe and Haar cascade face detection;
- MediaPipe FaceMesh landmarks for pose, eyes, and related checks;
- deterministic OpenCV/Pillow image measurements;
- a TorchVision Faster R-CNN tie detector;
- optional U2-Net/rembg background segmentation for `/edit-photo` only;
- deterministic policy aggregation in `verify.py`.

The design is broadly aligned with the proposed architecture: perception,
measurements, policy, and structured output are separated more than they would
be in a single opaque classifier. Replacing Flask, Faster R-CNN, or U2-Net is
not justified by the repository evidence alone.

The most important current findings are:

- The earlier `run_checks()` crash has been fixed in the current worktree.
- The PHP service now sends `X-Service-Token` to Python in the current
  worktree, but the shared secret must still be configured identically on both
  runtimes.
- The Python test suite currently passes 222 tests, with 2 skipped tests.
- A live Gunicorn smoke test previously verified `/ready`, `/verify`, and
  `/edit-photo` over HTTP.
- A custom university tie model is not present. The default effective backend
  is the generic TorchVision COCO detector when explicitly configured or when
  `auto` finds no custom checkpoint.
- No university-labelled dataset exists in the repository. No accuracy claim
  for the tie detector or the complete pipeline is justified.
- Python 3.9 dependency constraints leave known security advisories in Pillow,
  Torch, and rembg unresolved; a Python 3.11 or 3.12 migration is required for
  the next hardening cycle.
- Training capture exists in the current worktree but is disabled by default.
  It must remain disabled until governance approval. This audit does not enable
  it and recommends no automatic collection of university photographs.

**Testing-deployment decision:** suitable for a controlled internal testing
deployment after the deployment checklist is completed. Not approved by this
document for unrestricted student-facing production traffic.

**Evidence versus judgment:** statements beginning with “Current” or
“Verified” are based on repository code or commands run during this audit.
“Recommendation” and priority labels are engineering judgment informed by the
linked authoritative sources.

## 2. Current Architecture

### Request flow

```text
Student browser
    |
    v
PHP portal: upload, admin settings, SQLite persistence
    |
    | multipart POST + X-Service-Token
    v
Python Flask application
    |
    +--> validate/decode image
    +--> detect faces
    +--> execute enabled checks
    +--> aggregate pass-count and mandatory policies
    +--> return JSON
    |
    +--> optional /edit-photo call after a PHP-side pass
```

### Python entry points and modules

| File | Current responsibility | Evidence |
|---|---|---|
| `python-service/app.py` | Flask app, auth, CORS, upload handling, model warm-up, readiness, image editing, `/verify` | `app.py:63-707` |
| `python-service/verify.py` | Image decoding, face detection, quality checks, background analysis, tie policies, aggregation | `verify.py:49-1790` |
| `python-service/tie_detector.py` | Tie detector protocol, custom Faster R-CNN loader, COCO detector, policy/checksum validation | `tie_detector.py:51-590` |
| `python-service/tie_visibility.py` | Face-relative upper-body visibility and ROI calculation | `tie_visibility.py:34-157` |
| `python-service/training_capture.py` | Optional original-image and inference-metadata capture; disabled by default | `training_capture.py:21-107` |
| `python-service/training/dataset.py` | COCO loader and identity-disjoint split helper for offline training | `training/dataset.py:25-175` |
| `python-service/training/train_tie_detector.py` | Offline Faster R-CNN training script | `training/train_tie_detector.py` |
| `python-service/training/calibration.py` | Offline threshold calibration | `training/calibration.py` |
| `python-service/training/evaluate_tie_detector.py` | Offline evaluation script | `training/evaluate_tie_detector.py` |
| `python-service/gunicorn.conf.py` | Production WSGI process configuration | `gunicorn.conf.py:1-81` |
| `python-service/requirements.txt` | Pinned runtime dependency declarations | `requirements.txt:1-52` |

### PHP integration

`php-app/api/verify.php` constructs the multipart request and currently sends
`X-Service-Token` on both Python calls at `verify.php:71-105` and
`verify.php:142-153`. It reads `PORTAL_SHARED_SECRET` from the server
environment at `verify.php:75-80`.

PHP currently:

- reads enabled criteria and policy settings from its database;
- sends them as `criteria` and `params` JSON fields;
- stores the result JSON in SQLite;
- stores passed images after the optional `/edit-photo` transformation;
- deletes failed temporary uploads at `verify.php:170-174`;
- does not currently send `X-Training-Consent` or `X-Applicant-Identity`.

The last two points are important for future training governance. Failed
photos are not available for later review, and passed originals are not
preserved by this PHP flow.

## 3. Target Architecture

The proposed architecture is appropriate and can be reached incrementally
without a framework rewrite:

```text
University Portal
       |
       | authenticated multipart request over private HTTP
       v
+--------------------------+
| Upload/API Layer         |
| auth, limits, validation |
+------------+-------------+
             v
+--------------------------+
| Preprocessor             |
| orientation, decode,     |
| format and size checks   |
+------------+-------------+
             |
       +-----+------+
       v            v
  Face/person    Optional subject
  detection      segmentation
       |            |
       +-----+------+
             v
+--------------------------+
| Explainable measurements |
| tie, background, quality |
| framing and face checks  |
+------------+-------------+
             v
+--------------------------+
| Policy/rules engine      |
| institution configuration|
+------------+-------------+
             v
   PASS / FAIL / RETAKE / REVIEW
             |
             v
       Versioned JSON API
```

The existing code already has the main conceptual boundaries. The next
important boundary is to make model outputs normalized evidence and keep
institution-specific decisions in a policy layer rather than inside detector
implementations.

## 4. Current vs Recommended

| Area | Current | Recommended | Priority |
|---|---|---|---|
| API entry point | Flask routes `/verify`, `/edit-photo`, `/warmup`, `/ready`, `/health` | Keep Flask/Gunicorn for now; add a versioned API alias only during a planned compatibility release | P2 |
| Authentication | Shared `X-Service-Token`, constant-time comparison; PHP currently sends it | Keep internal-only exposure, rotate secret through deployment secret management, add endpoint authorization tests | P0/P1 |
| CORS | Disabled unless `PORTAL_CORS_ORIGINS` is explicitly set | Keep disabled for PHP server-to-server calls; allow only exact origins if browser calls Python directly | P1 |
| Upload size | Flask request cap plus decoded pixel cap | Also enforce limits at reverse proxy/PHP and test slow uploads | P0/P1 |
| Image formats | Decoded Pillow format allowlist: JPEG, PNG, WEBP | Keep allowlist and validate file signature at the edge as defense in depth | P0 |
| Model loading | Per-worker cached loading; custom policy validation; COCO fallback | Pre-stage all weights, verify hashes, fail readiness if required artifacts are missing | P0 |
| Tie model | Generic COCO class 32 unless calibrated custom model is supplied | Use COCO only as a documented baseline; promote a custom model only after labelled evaluation | P1 |
| Tie policy | Some detector-specific logic is mixed into `verify.py` | Normalize evidence, then evaluate policy separately | P1 |
| Background | Deterministic LAB/HSV/RGB border analysis; U2-Net used for editing | Keep deterministic validation; retain U2-Net only if editing is a business requirement | P1 |
| Training | Offline scripts plus disabled opt-in capture module | No live training; governed offline labelling, evaluation, approval, and rollback | P0/P2 |
| Data identity | `identity_id` required by current training split helper | Use pseudonymous application/admission references in offline datasets only | P1 |
| Readiness | Authenticated `/ready` warms and checks models | Use `/health` for liveness and `/ready` for traffic gating | P0 |
| Process server | Gunicorn configuration exists; two workers default | Measure memory on target host before selecting worker count | P0 |
| Dependencies | Direct versions pinned; Python 3.9 limitations documented | Move to Python 3.11/3.12 and regenerate a secure lock/constraints set | P0/P1 |
| Observability | Basic logging and timing metadata in responses | Add request IDs, latency/error metrics, saturation alerts, and redacted access logs | P1 |
| Deployment | No Docker, systemd, reverse-proxy, or CI artifact in repository | Add the one deployment artifact matching the university's actual host | P1 |

## 5. Computer Vision Assessment

### Tie detection

**Verified implementation**

- `CocoTieDetector` uses TorchVision Faster R-CNN ResNet-50-FPN V2 weights and
  COCO label 32 (`tie_detector.py:453-590`).
- `TorchTieDetector` expects a two-class custom checkpoint and uses
  `weights_only=True` when loading (`tie_detector.py:286-304`).
- A custom model requires an adjacent policy file with a SHA-256 binding,
  calibrated threshold, geometry restrictions, and held-out metrics
  (`tie_detector.py:84-162`).
- The upper-body ROI is face-relative (`tie_visibility.py:117-157`).
- Detection boxes are checked against face-relative width, height, vertical
  position, and horizontal alignment (`tie_detector.py:165-209`).
- Required-tie detection treats missing or ambiguous evidence as manual review
  rather than approval (`verify.py:1166-1269`).

**Assessment**

Faster R-CNN is technically appropriate for a high-accuracy object-detection
baseline when bounding boxes and a small number of classes matter. It is not
proven appropriate for this university population because no university
dataset or measured latency/memory budget exists. The model is also expensive
for CPU inference: TorchVision's official model table shows substantially
smaller MobileNet alternatives, but choosing one without measuring accuracy on
ties, partial occlusions, bow ties, scarves, and hard negatives would be
guessing.

The current COCO detector is not a university-specific tie model. A COCO “tie”
class should not be described as detecting every acceptable form of formal
neckwear. Bow ties, unusual ties, occluded ties, religious/cultural garments,
scarves, and tie-like clothing need explicit test examples.

**Important correctness risk**

`check_no_tie()` returns an automatic accepted `tie_absent` result when
`detector.detect()` returns `None` at `verify.py:1391-1405`. The COCO detector
declares `supports_absence_decision=True`, but the custom one-class detector
does not declare that capability. A custom-model miss can therefore fail open.

Recommendation: make absence approval capability explicit and fail closed to
manual review for detectors that cannot establish absence. **Priority: P0 if
the custom backend is ever enabled; P1 while only COCO is allowed.**

### Background detection and segmentation

The white-background criterion is deterministic. It samples a masked border,
uses HSV/RGB/LAB measurements, coverage gates, contamination guards, and
luminance uniformity. The preset policy is in `verify.py` around
`BACKGROUND_STRICTNESS_LEVELS` and `_resolve_background_params`.

This is a good explainability choice. The repository tests specifically cover
white clothing, shadows, grey backgrounds, coloured backgrounds, off-white
walls, and texture. Those tests do not establish real-world accuracy because
many real fixtures use developer-specific absolute paths and are skipped when
absent.

U2-Net/rembg is used for subject cutouts and background repainting in
`app.py:189-398`, not as the core white-background validator. It is therefore
not required for the validation decision itself. Keeping it optional reduces
startup cost and attack surface if image editing is not required.

Known cases requiring measured validation:

- white shirts against white walls;
- bright lighting and clipped highlights;
- cream and warm-grey walls;
- shadows and textured walls;
- patterned backgrounds;
- incomplete subject masks;
- no detected face.

Do not change thresholds based on intuition. Calibrate them against labelled
images and record false acceptance and false rejection rates.

### Person, face, and framing

MediaPipe Face Detection is preferred, with OpenCV Haar fallback at
`verify.py:184-239`. FaceMesh supports pose and eye-related checks. The code
uses the first detected face for several downstream checks, so multiple-face
handling depends on the face-count criterion being enabled.

Recommendations:

- Keep face count as a mandatory policy for this use case.
- Ensure downstream checks never approve an image when the selected face is
  ambiguous or the required landmarks are unavailable.
- Measure false negatives for face coverings, lighting, skin tones, camera
  quality, and head pose.
- Avoid gender or demographic inference. Tie applicability must come from an
  explicit university policy, not from the image.

## 6. Security Assessment

### Strengths currently present

- `MAX_CONTENT_LENGTH` limits request body size (`app.py:65-66`).
- Decoded pixel count is bounded and Pillow decompression-bomb protection is
  enabled (`verify.py:77-124`).
- The decoded format is restricted to JPEG, PNG, and WEBP.
- `/verify`, `/edit-photo`, `/warmup`, and `/ready` require the service token.
- Secret comparison uses `hmac.compare_digest` (`app.py:131-134`), consistent
  with the Python standard-library guidance.
- Resize width, height, and total output pixels are bounded
  (`app.py:500-518`).
- Errors returned by the Python API do not expose stack traces or raw internal
  exceptions to callers.
- The custom model policy binds the checkpoint to a SHA-256 digest.
- Uploaded images are not logged by the Python code.

### Remaining risks

**P0 - deployment network boundary**

Python must remain on loopback or a private, firewall-restricted interface.
The default Gunicorn binding is loopback (`gunicorn.conf.py:27-30`). If PHP and
Python are placed on separate hosts, use private networking, firewall rules,
and HTTPS. A shared static token alone is not a complete network boundary.

**P0 - rate limiting and resource governance**

There is no application rate limiter, per-client quota, reverse-proxy request
rate policy, or bounded queue. Inference is CPU- and memory-intensive. Gunicorn
timeouts and upload limits help but do not prevent an authenticated caller
from consuming all workers. Apply rate limiting and process/container resource
limits at the actual edge or host.

Source: [OWASP API4:2023 Unrestricted Resource Consumption](https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/)
recommends limits for execution time, memory, upload size, operations, and
request frequency.

**P1 - PHP storage permissions and public uploads**

The PHP code stores passed images under `php-app/uploads` and creates the
directory with `0750` after the earlier hardening change. Confirm the web
server cannot execute uploaded content and that direct public retrieval is
intentional. Prefer storage outside the web root with an authenticated
download handler.

Source: [OWASP File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)
recommends generated server-side names, allowlists, size limits, authorization,
and storage outside the web root where possible.

**P1 - dependency vulnerabilities**

The dependency file is pinned, but the current Python 3.9 runtime prevents
moving all packages to their patched releases. A previous `pip-audit` run on
the Python 3.9 environment reported advisories in Pillow 11.3.0, Torch 2.8.0,
rembg 2.0.61, and transitive dependencies. That audit was run before the
Flask/Flask-CORS update and must be rerun from a clean environment before
release. Do not represent the current dependency set as vulnerability-free.

**P1 - supply-chain and model downloads**

TorchVision and rembg can download model artifacts when their caches are empty.
The deployment must pre-stage, checksum, and permission model files. A live
student request must never be the event that downloads a model.

## 7. Privacy / Data Governance

Student photographs are personal data and may be biometric or otherwise
sensitive under applicable university policy and law.

The required safe state is:

```text
TRAINING_CAPTURE_ENABLED=0
```

The current `training_capture.py` is opt-in and the application only invokes
it when the environment flag is enabled and `X-Training-Consent` is present
(`app.py:612-629`). The current PHP code does not send that header or an
applicant identity header. This audit does not enable capture, add identity
storage, or recommend automatic dataset creation.

The capture module must remain disabled until the university approves:

- a separate training-use consent or lawful basis;
- retention and deletion periods;
- access roles and audit logging;
- encryption and backup policy;
- data minimization and pseudonymous application references;
- human labelling procedures;
- withdrawal/opt-out handling;
- dataset poisoning and quality controls.

Future training, if approved, must be offline and human-reviewed. Never use a
model's own prediction as its ground-truth label. Never train or replace the
live model during a student request.

## 8. API Recommendation

### Keep the current contract for the immediate release

The current PHP application expects form fields and the existing JSON shape.
Do not redesign the public contract immediately before deployment.

Current endpoints:

| Endpoint | Method | Auth | Current purpose |
|---|---:|---:|---|
| `/health` | GET | No | Process liveness and check list |
| `/ready` | GET | Yes | Model readiness for traffic gating |
| `/warmup` | GET/POST | Yes | Explicit model warm-up |
| `/verify` | POST | Yes | Multipart image verification |
| `/edit-photo` | POST | Yes | Multipart image transformation |

### Future versioned contract

For reuse by admissions, library cards, staff IDs, and other university
systems, introduce a new versioned endpoint only with a compatibility plan:

```http
POST /v1/photo/validate
Content-Type: multipart/form-data
X-Service-Token: ...
```

Recommended response shape:

```json
{
  "schema_version": 1,
  "decision": "PASS",
  "request_id": "server-generated-id",
  "model_versions": {
    "tie": "torchvision-fasterrcnn-resnet50-fpn-v2-coco"
  },
  "checks": {
    "person": {"passed": true, "status": "pass"},
    "tie": {"passed": true, "status": "pass"},
    "background": {"passed": true, "status": "pass"},
    "quality": {"passed": true, "status": "pass"},
    "framing": {"passed": true, "status": "pass"}
  },
  "reasons": []
}
```

Stable reason codes should be machine-readable, for example:

```text
IMAGE_INVALID
IMAGE_TOO_LARGE
IMAGE_TOO_SMALL
IMAGE_TOO_BLURRY
PERSON_NOT_DETECTED
MULTIPLE_PEOPLE
TIE_NOT_DETECTED
BACKGROUND_NOT_WHITE
FACE_NOT_VISIBLE
FACE_TOO_SMALL
BAD_FRAMING
REVIEW_REQUIRED
```

Human-facing messages can change without breaking consuming systems.

## 9. Policy / Rules Engine

The desired boundary is:

```text
model output
    |
    v
normalized evidence and confidence
    |
    v
university policy
    |
    v
PASS / FAIL / RETAKE / MANUAL_REVIEW
```

The current `run_checks()` function combines execution and aggregation at
`verify.py:1673-1790`. It already has explicit mandatory criteria and a
pass-count gate, but the policy is still coupled to the check registry and
parameter dictionaries.

Recommendations:

- Keep current behavior for the immediate release.
- Define an explicit policy object in a future version.
- Keep tie applicability separate from gender inference. For people who are
  not required to wear ties, policy should disable tie requirements explicitly.
- Do not allow arbitrary client parameters to override administrator policy.
- Record policy version and model version with each result.

Current threshold examples include:

- `min_pass_criteria` default `4` in `verify.py:1752-1756`;
- background presets in `verify.py` around `BACKGROUND_STRICTNESS_LEVELS`;
- COCO tie positive threshold default `0.50` in `tie_detector.py:467`;
- no-tie thresholds default `0.65` reject and `0.30` accept in
  `verify.py:1279-1285`;
- upper-body visibility defaults of face height `80` and below-face ratio
  `0.35` in `tie_visibility.py:13-17,88-94`.

These are implementation values, not validated university policy. Do not
change them without a representative labelled evaluation set.

## 10. Confidence Bands

The current tie checks implement a confidence band in places:

- high confidence can accept required-tie presence or reject no-tie;
- low confidence can accept no-tie only for a detector capable of an absence
  decision;
- intermediate confidence routes to manual review or CV corroboration.

The numerical thresholds are not proven calibration for this university.

Recommended calibration methodology:

1. Define the intended policy and error costs with university stakeholders.
2. Collect a representative, governed dataset outside the live inference path.
3. Split by applicant identity, not by image.
4. Keep calibration data separate from the held-out test set.
5. Measure precision, recall, F1, false acceptance, false rejection, and
   review rate.
6. Choose thresholds against explicit error-cost targets.
7. Report confidence intervals and sample counts.
8. Test hard negatives and edge cases before promotion.

No accuracy or threshold-quality claim is made in this document.

## 11. Testing Strategy

### Current verification

The last full run on the current implementation reported:

```text
222 tests passed
2 tests skipped
0 tests failed
```

The skipped tests depend on external developer-specific image paths. That
means the suite is useful regression coverage but is not a complete portable
acceptance suite.

### Unit tests

Maintain and extend coverage for:

- image format, signature, pixel, and decompression-bomb handling;
- invalid JSON and numeric parameter handling;
- background measurements and every strictness preset;
- tie geometry and detector absence capability;
- face visibility and malformed face boxes;
- policy aggregation, mandatory criteria, strict mode, and pass-count mode;
- stable error codes and no internal exception disclosure;
- model policy checksum and threshold validation.

### Integration tests

Required scenarios:

- PHP cURL to Gunicorn with correct token;
- missing, wrong, and rotated token;
- valid JPEG/PNG/WEBP upload;
- corrupt and mislabeled upload;
- oversized body and oversized decoded image;
- multiple concurrent requests;
- model warm-up and worker restart;
- `/ready` failure when required weights are unavailable;
- `/edit-photo` success and bounded resize failure;
- reverse proxy timeout and upstream failure;
- PHP handling of every Python HTTP status and JSON error shape.

### CV golden dataset

Create a governed dataset specification, without collecting it automatically
from production:

Positive examples:

- plain white backgrounds;
- valid ties of different colours and patterns;
- different cameras, resolutions, lighting, and subjects;
- varying face sizes and poses;
- accepted cultural or institutional attire categories.

Negative examples:

- no tie;
- bow tie, scarf, necklace, lanyard, and tie-like objects;
- partially hidden or occluded tie;
- white, cream, grey, coloured, textured, and shadowed backgrounds;
- white clothing against white backgrounds;
- multiple people;
- blurry, corrupted, low-resolution, and poorly framed images.

Report precision, recall, F1, false acceptance, false rejection, manual-review
rate, and subgroup error analysis where lawful and appropriate. Do not claim
actual performance until these measurements exist.

## 12. Performance Recommendation

The pipeline is synchronous and CPU-heavy. A Gunicorn worker can load its own
Torch, ONNX, and MediaPipe state. Therefore `GUNICORN_WORKERS=2` can roughly
double model memory; it must not be accepted as safe without target-host
measurement.

Measure on the university host:

- model load and warm-up time;
- cold and warm p50, p95, and p99 latency;
- resident memory per worker and total memory;
- CPU saturation and inference concurrency;
- request queueing and timeout behavior;
- worker recycling behavior;
- upload/decode time separately from model time.

Current Gunicorn choices are reasonable for a first CPU pilot:

- sync workers;
- one thread per worker;
- `preload_app=False`;
- timeout `120` seconds;
- graceful timeout `30` seconds;
- bounded request fields;
- max requests `200` plus jitter;
- model warm-up after each worker starts.

Do not migrate to FastAPI/ASGI merely for fashion. The workload is blocking
CPU inference, and the existing Flask WSGI boundary is sufficient for the
current synchronous use case.

ONNX Runtime documents that session thread pools and intra/inter-op settings
must be measured and tuned for the actual workload. Avoid multiplying Gunicorn
workers and native model threads until CPU contention is measured.

## 13. MLOps Recommendation

Use a lightweight lifecycle:

```text
Governed data collection, if approved
        |
        v
Human annotation and quality review
        |
        v
Identity-disjoint train/validation/test split
        |
        v
Training and calibration
        |
        v
Held-out evaluation and error analysis
        |
        v
Human approval and artifact checksum
        |
        v
Staging deployment and smoke/load tests
        |
        v
Production promotion with rollback
        |
        v
Operational and model monitoring
```

Minimum artifact manifest:

- model version;
- model SHA-256;
- code revision;
- dependency lock identifier;
- dataset version and permitted provenance;
- training configuration and random seed;
- calibration data identifier;
- held-out metrics;
- policy version;
- deployment timestamp and approver.

Monitor:

- request count and error rate;
- p50/p95/p99 latency;
- timeout and readiness failures;
- worker memory and CPU;
- decision distribution and manual-review rate;
- input quality distributions;
- drift against approved reference data;
- labelled performance when lawful ground truth becomes available.

Microsoft's guidance emphasizes lineage, model/data monitoring, drift signals,
and human-gated promotion. This project does not need Azure services to adopt
those controls; files, scripts, a secure server, and an approval record are a
proportionate first step.

## 14. P0 Changes

P0 means required before a controlled testing deployment can be trusted.

- Confirm the current code and tests are the exact revision being deployed.
- Set the same strong `PORTAL_SHARED_SECRET` in PHP and Python secret stores.
- Keep Python private; expose only PHP publicly.
- Pre-stage and checksum TorchVision and U2-Net model artifacts.
- Set `TIE_DETECTOR_BACKEND` explicitly; do not rely on `auto` in production.
- Gate traffic on authenticated `/ready` after model warm-up.
- Measure target-host memory before selecting two workers.
- Apply reverse-proxy or host-level rate limiting and resource limits.
- Verify HTTPS at the public edge and safe internal transport if hosts differ.
- Keep `TRAINING_CAPTURE_ENABLED=0`.
- Do not enable a custom tie model without its policy file and evaluation.
- If the custom backend will be enabled, fix the custom-detector no-detection
  fail-open behavior before release.

## 15. P1 Changes

- Upgrade from Python 3.9 to 3.11 or 3.12 and regenerate dependencies.
- Run a clean-environment dependency audit and maintain an SBOM or equivalent
  dependency inventory.
- Add portable fixtures instead of developer-specific absolute test paths.
- Add CI for tests, linting, syntax, dependency audit, and deployment checks.
- Add request IDs and metrics without logging image contents or secrets.
- Add PHP/Python integration tests to the CI environment.
- Validate public upload storage and web-server execution rules.
- Add a formal policy object and policy version to responses.
- Measure repeated MediaPipe/FaceMesh work and optimize only after profiling.
- Add a systemd unit, container definition, or equivalent artifact matching the
  actual university hosting platform.
- Add a rollback runbook for code, dependencies, and model artifacts.

## 16. P2 Changes

- Add `/v1/photo/validate` with a compatibility adapter for the existing PHP
  form contract.
- Replace ad hoc dictionaries with typed request/result schemas.
- Add a dedicated human-review workflow if the university requires it.
- Evaluate a smaller detector only after the golden dataset supplies accuracy
  and latency evidence.
- Add drift dashboards and scheduled offline evaluation.
- Add approved, governed training-data workflows in a separate system.

## 17. Not Recommended

- Do not train or replace the live model inside a student request.
- Do not enable `TRAINING_CAPTURE_ENABLED` without governance approval.
- Do not collect uploaded university photographs automatically.
- Do not infer gender or tie applicability from a face image.
- Do not replace Faster R-CNN merely because another detector is newer.
- Do not replace Flask with FastAPI without a measured requirement.
- Do not replace U2-Net if background editing is still required; remove it only
  if that feature is intentionally retired.
- Do not silently change confidence thresholds.
- Do not use model predictions as ground-truth labels.
- Do not use random image-level splits when multiple images may belong to one
  applicant.
- Do not expose Python directly to the public internet.
- Do not deploy with empty model caches and depend on live downloads.
- Do not claim accuracy without a representative labelled dataset.

## 18. Proposed Implementation Plan

This is a recommendation plan, not executed by this audit:

```text
0:00-0:30  Freeze revision, inspect environment, confirm governance flags
0:30-1:15  Verify private network, secrets, model artifacts, and readiness
1:15-2:00  Run unit/integration tests and remove non-portable test dependence
2:00-2:45  Run target-host warm-up and memory/latency smoke measurements
2:45-3:30  Configure rate limits, reverse proxy, process supervision, logging
3:30-4:15  Run malformed-upload, concurrency, timeout, and worker-restart tests
4:15-4:45  Run dependency audit and record unresolved advisories
4:45-5:00  Review rollback checklist and obtain human deployment approval
```

If a defect is found during this sequence, stop the release rather than
silently weakening a policy or skipping a failed test.

## 19. Research Sources

### Authoritative sources used

- [Flask deployment documentation](https://flask.palletsprojects.com/en/stable/deploying/)
  says the built-in development server is not designed for production and a
  dedicated WSGI server should be used.
- [Flask configuration documentation](https://flask.palletsprojects.com/en/stable/config/)
  documents `MAX_CONTENT_LENGTH` and multipart form limits.
- [OWASP File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)
  recommends allowlists, server-side validation, size limits, generated names,
  authorization, and storage outside the web root where possible.
- [OWASP API4:2023 Unrestricted Resource Consumption](https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/)
  recommends limiting execution time, memory, upload size, operations, and
  request frequency.
- [OWASP REST Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)
  recommends HTTPS and endpoint-level access control for non-public APIs.
- [Pillow `Image.open` documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open)
  documents decompression-bomb warnings/errors and `MAX_IMAGE_PIXELS`.
- [Python `hmac` documentation](https://docs.python.org/3/library/hmac.html)
  recommends `compare_digest()` for secret comparisons.
- [Gunicorn settings documentation](https://gunicorn.org/reference/settings/)
  documents worker timeouts, graceful timeouts, worker recycling, and preload
  behavior.
- [Python Packaging reproducible environments](https://packaging.python.org/en/latest/specifications/section-reproducible-environments/)
  provides guidance for reproducible dependency environments.
- [Microsoft Azure Architecture Center MLOps](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/machine-learning-operations-v2)
  describes monitoring, drift, infrastructure signals, and human-in-the-loop
  lifecycle management. Only the lifecycle principles are recommended here;
  no Azure service is required.
- [Microsoft model monitoring guidance](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2)
  recommends monitoring production input/output signals and using appropriate
  reference data.
- [NIST AI Risk Management Framework](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)
  describes trustworthy AI characteristics including validity, reliability,
  safety, security, accountability, explainability, privacy, and fairness.
- [TorchVision model documentation](https://docs.pytorch.org/vision/stable/models.html)
  documents available object-detection architectures and model-weight
  characteristics. It does not establish that any listed model is accurate for
  this university's tie policy.
- [Google MediaPipe Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
  documents landmark detection modes and the fact that synchronous image
  detection blocks the calling thread.
- [OpenCV thresholding documentation](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html)
  documents deterministic thresholding methods relevant to explainable image
  measurements.
- [ONNX Runtime thread-management documentation](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
  documents intra-op/inter-op thread configuration and the need to tune it for
  the workload.

### YouTube videos

No YouTube videos were used as research evidence.

## 20. Approval Gate

```text
IMPLEMENTATION STATUS: NOT APPROVED

No source-code changes should be made from this research until
the human owner reviews and explicitly approves the P0/P1 plan.
```
