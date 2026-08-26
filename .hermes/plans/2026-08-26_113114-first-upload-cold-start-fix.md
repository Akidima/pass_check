# First-Upload Cold-Start Fix Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Eliminate the ~17 s first-upload penalty so every student gets a validated photo in under ~5 seconds, even on the very first request after the Python service restarts.

**Architecture:** Both ML models (torchvision Faster R-CNN tie detector, rembg/U2-Net background segmenter) are lazily initialized and cached per worker process. We add a `/warmup` endpoint that idempotently triggers both loads, and call it automatically in a background thread at service startup — moving the cost to boot time where no browser is waiting. No restructuring; three small additive edits in `python-service/app.py`, one new test file, one doc sentence.

**Tech Stack:** Flask 3.0, functools `@lru_cache`, threading (stdlib), rembg/onnxruntime, torchvision. Existing test runners: Python unittest (`python-service/tests/`), PHP (`tests/test_background_settings.php`).

---

## Current context / assumptions (verified by experiment, Aug 26 2026)

Measured breakdown of the observed 16.8 s first upload vs ~4.2 s warm:

| Stage | Cold | Warm | Where |
|---|---|---|---|
| `POST /verify` — tie detector lazy load (`@lru_cache`) | ~5.7 s | ~0 s | `python-service/tie_detector.py:498-509` |
| Hidden 2nd call `POST /edit-photo` (only when photo PASSES) — U2-Net lazy load | **~10.6 s** | ~0.5 s | `php-app/api/verify.php:124-149` → `python-service/app.py:42-54` |
| Upload transfer + PHP glue + SQLite insert | <0.3 s | <0.1 s | `php-app/api/verify.php` |

- The service self-reports only its own `/verify` internals in `timings_ms`; the `/edit-photo` round trip happens later inside PHP and is invisible there.
- `models/tie_detector_v1.pt` does NOT exist; backend resolves to COCO fallback weights (`~/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`, 175 MB).
- `~/.cache/u2net/` did not exist before testing — first `/edit-photo` downloads/initializes the ONNX model.
- Service currently runs via `cd python-service && ./venv/bin/python app.py` (dev server, `debug=True`, port 5001).
- Python suite baseline: 175 tests OK (2 skipped). PHP suite: 14 checks, 0 failures.

### ELI5: why the first photo is slow

Imagine the workshop keeps its two big toys **in a locked closet** and only unpacks them when someone first asks to play. The first kid to upload a photo waits while we unpack Toy #1 (tie-spotter, ~6 s) and Toy #2 (background-painter, ~11 s) — one after the other. After that, the toys stay on the table, so everyone else is quick. **The fix: unpack both toys the moment the workshop opens (at startup), before any kid knocks.** And because unpacked toys stay out all day, nobody ever waits again.

---

## Research references (authentic links used for these fixes)

- Flask routing & `jsonify`: https://flask.palletsprojects.com/en/3.0.x/api/#flask.Flask.route
- `functools.lru_cache` semantics (why reload cost is paid exactly once per worker): https://docs.python.org/3/library/functools.html#functools.lru_cache
- `threading.Thread(daemon=True)` for non-blocking startup: https://docs.python.org/3/library/threading.html#threading.Thread
- torchvision Faster R-CNN ResNet50-FPN V2 (COCO weights auto-download/caching): https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
- rembg `new_session('u2net')` (lazy ONNX session): https://github.com/danielgatis/rembg
- onnxruntime session initialization cost: https://onnxruntime.ai/docs/get-started/with-python.html
- Local evidence: measured timings recorded in this session (Stage A/B/C experiment, 2026-08-26).

---

## Files likely to change

| File | Action |
|---|---|
| `python-service/app.py` | Modify line 20 (extend import), insert helper + route after line 54, replace lines 457-459 (startup warm-up) |
| `python-service/tests/test_warmup_endpoint.py` | Create |
| `README.md` | Modify lines 129-132 (one sentence) |

No other files touched. No restructuring. Total production diff ≈ 30 added lines, 1 modified import line.

---

## Proposed approach (chosen over alternatives)

**Option A — chosen: pre-warm at startup + explicit `/warmup` endpoint.**
Cost moves to service boot (operator-visible, harmless), students never wait. Endpoint doubles as a manual/CI warm-up trigger and makes the behavior observable/testable. Smallest diff, zero UX-semantics change, DRY (one `_warm_all_models()` used by both callers).

**Option B — rejected (YAGNI): make the `/edit-photo` storage step asynchronous in `api/verify.php`.**
Would respond faster but changes semantics (student sees "Approved" before the stored copy is repainted white), requires job tracking/retry, touches PHP significantly. Unnecessary once models are pre-warmed: warm `/edit-photo` is only ~0.5 s.

**Option C — rejected: eager import-time model loading (module top level).**
Blocks server bind for ~13-17 s; health checks fail during deploys; harder to test. Background thread achieves the same without the outage window.

---

## Step-by-step plan

### Task 1: Add `_warm_all_models()` helper and `/warmup` endpoint

**Objective:** One idempotent function loads both lazy models; an HTTP route exposes it.

**Files:**
- Modify: `python-service/app.py:20` (import line)
- Modify: `python-service/app.py` (insert after line 54, i.e., directly below the closing `return REMBG_SESSION` of `get_rembg_session`)
- Test: `python-service/tests/test_warmup_endpoint.py` (create)

**Step 1: Write failing test**

Create `python-service/tests/test_warmup_endpoint.py` with exactly:

```python
"""Tests for the /warmup endpoint that pre-loads lazy ML models.

The endpoint must be idempotent, cheap when already warm, survive a broken
detector backend (report the error, still answer 200), and accept GET+POST.
"""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import app as service_app  # noqa: E402


class WarmupEndpointTests(unittest.TestCase):
    def setUp(self):
        service_app.app.config["TESTING"] = True
        self.client = service_app.app.test_client()

    def test_warmup_returns_ok_and_reports_loaded_models(self):
        with patch("app.get_tie_detector") as fake_det, \
             patch("app.get_rembg_session", return_value=object()):
            fake_det.return_value = object()
            resp = self.client.post("/warmup")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["tie_detector_loaded"])
        self.assertTrue(data["rembg_session_loaded"])

    def test_warmup_survives_detector_failure(self):
        with patch("app.get_tie_detector", side_effect=RuntimeError("boom")), \
             patch("app.get_rembg_session", return_value=None):
            resp = self.client.post("/warmup")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["tie_detector_loaded"])
        self.assertIn("tie_detector_error", data)
        self.assertFalse(data["rembg_session_loaded"])

    def test_warmup_accepts_get_too(self):
        with patch("app.get_tie_detector"), \
             patch("app.get_rembg_session", return_value=object()):
            resp = self.client.get("/warmup")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify failure**

Run: `python-service/venv/bin/python -m unittest discover -s python-service/tests -p test_warmup_endpoint.py -v`
Expected: FAIL/ERROR — 404 on `/warmup` (route does not exist yet).

**Step 3: Implement (three small edits)**

Edit 1 — extend the existing import at `python-service/app.py:20`:

```python
from verify import run_checks, CHECK_LABELS, _detect_faces, get_tie_detector
```

Edit 2 — insert directly after line 54 (`return REMBG_SESSION`), leaving one blank line either side:

```python
def _warm_all_models():
    """Load both lazily-initialized ML models once so first requests are fast.

    Idempotent: the tie detector is cached via ``@lru_cache`` in
    tie_detector.py and the rembg session via the module globals guarded in
    get_rembg_session(). Safe to call from the startup thread and /warmup.
    """
    result = {}
    try:
        get_tie_detector()
        result["tie_detector_loaded"] = True
    except Exception as exc:  # never let warm-up crash the caller
        result["tie_detector_loaded"] = False
        result["tie_detector_error"] = str(exc)[:200]
    result["rembg_session_loaded"] = get_rembg_session() is not None
    return result


@app.route("/warmup", methods=["GET", "POST"])
def warmup():
    """Pre-load lazy ML models. Cheap/no-op when already warm."""
    return jsonify({"status": "ok", **_warm_all_models()})
```

**Why (ELI5):** `_warm_all_models` is "unpack both toys." The try/except means a broken toy never stops the shop from opening — it just writes a note saying that toy failed. `@lru_cache` means "once built, keep it on the table," so calling it again later costs nothing.

**Step 4: Run test to verify pass**

Run: `python-service/venv/bin/python -m unittest discover -s python-service/tests -p test_warmup_endpoint.py -v`
Expected: PASS — `Ran 3 tests ... OK`.

**Step 5: Commit**

```bash
git add python-service/app.py python-service/tests/test_warmup_endpoint.py
git commit -m "feat(service): add idempotent /warmup endpoint for lazy ML models"
```

---

### Task 2: Auto-warm at service startup (background thread)

**Objective:** Students never pay the model-load cost; the server still binds instantly.

**Files:**
- Modify: `python-service/app.py:457-459` (`if __name__ == "__main__":` block)

**Step 1: Apply the change** — replace the whole bottom block with:

```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))

    # Pre-load the lazy ML models in a background thread so the server starts
    # listening immediately and the first student upload never pays the
    # one-time model-load cost (~6 s tie detector + ~10 s U2-Net).
    threading.Thread(target=_warm_all_models, name="model-warmup", daemon=True).start()

    app.run(host="127.0.0.1", port=port, debug=True)
```

(`threading` is already imported at `python-service/app.py:14`.)

**Why (ELI5):** The moment the workshop door opens, one helper goes to the closet and unpacks the toys **while the door stays open**. `daemon=True` means if the shop closes early, the helper quietly leaves too — he never blocks closing time.

**Step 2: Sanity-run the full Python suite**

Run: `python-service/venv/bin/python -m unittest discover -s python-service/tests -v`
Expected: PASS — previously `Ran 175 tests ... OK (skipped=2)`; now 178 total, all OK. (Importing `app` in tests must NOT load models — the thread only starts under `__main__`.)

**Step 3: Commit**

```bash
git add python-service/app.py
git commit -m "feat(service): warm lazy ML models in background thread at startup"
```

---

### Task 3: Document reality in the README

**Objective:** Correct the claim that U2-Net never delays users, and document `/warmup`.

**Files:**
- Modify: `README.md:129-132` (final sentence of that paragraph)

**Step 1: Replace the sentence**

Old (`README.md:131-132`):

```
it does not delay photo validation.
```

New:

```
Both heavy models (tie detector, U2-Net/rembg) are pre-loaded in a background
thread at service startup and can be warmed manually via `POST/GET /warmup`;
a passing photo's stored copy is repainted via a synchronous `/edit-photo`
call in `php-app/api/verify.php`, which is fast (<1 s) once models are warm.
```

**Step 2: Verify nothing broke**

Run: `php tests/test_background_settings.php` (needs `PASS_CHECK_DB_PATH` env set first — see repo convention)
Expected: `14 checks, 0 failures` (unchanged; docs-only edit).

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe model pre-warming and the hidden edit-photo round trip"
```

---

### Task 4: End-to-end timing verification (proof)

**Objective:** Demonstrate first-upload is now <~5 s after warm-up completes.

**Steps (exact commands):**

```bash
# 1. Stop any running service, start fresh
cd python-service && ./venv/bin/python app.py &   # or your normal launcher

# 2. Warm it explicitly (first call blocks ~13-17 s, reports both flags true)
time curl -s -X POST http://127.0.0.1:5001/warmup
# Expected: {"rembg_session_loaded":true,"status":"ok","tie_detector_loaded":true}

# 3. FIRST real upload through the site must now be fast
curl -s -o /dev/null -w "%{time_total}s\n" \
  -F "photo=@<some-real-photo.jpg>;type=image/jpeg" \
  -F "student_name=Warm Proof" -F "student_id=WARM1" -F "source=upload" \
  http://localhost:8080/api/verify.php
# Expected: < 5.5s  (vs 16.8s before)

# 4. Regression suites
python-service/venv/bin/python -m unittest discover -s python-service/tests -v   # OK
php tests/test_background_settings.php                                           # 14/14
```

Success criteria: step 3 under ~5.5 s on the first post-restart upload; all suites green; `/health` still answers during warm-up (server bound immediately).

---

## Tests / validation summary

- New unit tests: 3 cases in `python-service/tests/test_warmup_endpoint.py` (mocked — no 175 MB downloads in CI/unit runs).
- Full regression: `python-service/venv/bin/python -m unittest discover -s python-service/tests -v`.
- PHP side untouched; suite re-run proves it.
- Manual E2E timing proof per Task 4 (cold-start scenario reproduced twice: before = 16.8 s, after = target <5.5 s).

## Risks, tradeoffs, open questions

- **Double-load race (minor, accepted):** `@lru_cache` is not a cross-thread lock — if a `/verify` request arrives *while* the startup thread is mid-load, torch may load twice once, wasting RAM/CPU briefly. If ever problematic, wrap the detector acquisition in a `threading.Lock`; deliberately omitted now (YAGNI).
- **Boot RAM grows earlier** (~400 MB models resident from t≈0 instead of t≈first-use). Irrelevant for this single-purpose service; relevant if the service later shares a tiny VPS with other tenants.
- **`debug=True` reloader** spawns a child process; the warm-up thread then runs once per process — idempotent by design, just slightly more boot CPU.
- **Failed photos were always fast** (they skip `/edit-photo`); this fix mainly benefits passing photos, which are the happy path.
- **Not done (documented decision):** async `/edit-photo` in PHP (Option B) — bigger semantic change, unnecessary after pre-warming.
