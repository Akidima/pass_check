<?php
require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';
require_admin();

$pdo = get_db();

// Admin panel displays photos/submissions that passed the criteria threshold
$sql = "SELECT * FROM submissions WHERE overall_passed = 1 ORDER BY created_at DESC LIMIT 200";
$submissions = $pdo->query($sql)->fetchAll(PDO::FETCH_ASSOC);

$pageTitle = 'Submissions';
include __DIR__ . '/../includes/header.php';
?>

<div class="topbar">
  <div class="brand">
    <div class="mark">🛡️</div>
    <div>Submissions Log <span class="sub">Review approved student photos</span></div>
  </div>
  <div class="nav-links">
    <a href="dashboard.php" class="pill-link">Dashboard</a>
    <a href="submissions.php" class="pill-link active">Submissions</a>
    <a href="settings.php" class="pill-link">Settings</a>
    <a href="logout.php" class="pill-link" style="color:#ff9d9d;">Logout</a>
  </div>
</div>

<div class="shell">
  <div class="card panel">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; flex-wrap:wrap; gap:12px;">
      <div>
        <h3 class="section-title" style="margin-bottom:2px;">Passed Submissions</h3>
        <p class="section-desc" style="margin:0;">Showing <?= count($submissions) ?> approved photo(s)</p>
      </div>
      <div class="nav-links">
        <span class="pill-link active">Passed Submissions</span>
      </div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Photo</th>
            <th>Student</th>
            <th>Source</th>
            <th>Status</th>
            <th>Date</th>
            <th style="text-align:right;">Actions</th>
          </tr>
        </thead>
        <tbody>
          <?php foreach ($submissions as $s): ?>
            <?php $result = json_decode($s['result_json'], true); ?>
            <tr id="row-<?= (int)$s['id'] ?>">
              <td>
                <img id="thumb-<?= (int)$s['id'] ?>" src="/uploads/<?= htmlspecialchars($s['filename']) ?>"
                     class="submission-thumb" alt="">
              </td>
              <td>
                <strong><?= htmlspecialchars($s['student_name'] ?: '—') ?></strong><br>
                <span style="color:var(--ink-500); font-size:12px;"><?= htmlspecialchars($s['student_id'] ?: '') ?></span>
              </td>
              <td><?= htmlspecialchars(ucfirst($s['source'])) ?></td>
              <td>
                <span class="badge pass">✓ Passed</span>
                <?php if (isset($result['passed_count']) && isset($result['total_criteria'])): ?>
                  <span style="font-size:11px; color:var(--ink-500); display:block; margin-top:2px;">
                    (<?= (int)$result['passed_count'] ?>/<?= (int)$result['total_criteria'] ?> criteria)
                  </span>
                <?php endif; ?>
              </td>
              <td><?= htmlspecialchars($s['created_at']) ?></td>
              <td style="text-align:right;">
                <button type="button" class="btn btn-outline edit-btn" style="padding:6px 14px; font-size:13px;"
                        data-id="<?= (int)$s['id'] ?>"
                        data-filename="<?= htmlspecialchars($s['filename']) ?>"
                        data-name="<?= htmlspecialchars($s['student_name'] ?: 'Student') ?>">
                  ✏️ Edit Photo
                </button>
              </td>
            </tr>
          <?php endforeach; ?>
          <?php if (empty($submissions)): ?>
            <tr><td colspan="6" style="text-align:center; color:var(--ink-500); padding:30px;">No passed submissions yet.</td></tr>
          <?php endif; ?>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- Edit Photo Modal -->
<div id="editModal" class="modal-overlay hidden">
  <div class="modal-card">
    <div class="modal-header">
      <h3>Edit & Enhance Photo</h3>
      <button type="button" id="closeModalBtn" class="close-btn">&times;</button>
    </div>
    <div class="modal-body">
      <div class="edit-grid">
        <!-- Live Preview Canvas / Image -->
        <div class="preview-box">
          <div class="preview-label">Passport Photo Preview</div>
          <div class="passport-frame">
            <div class="img-container">
              <canvas id="editorCanvas"></canvas>
              <img id="editorImage" src="" style="display:none;" alt="Photo source">
            </div>
          </div>
        </div>

        <!-- Controls -->
        <div class="controls-box">
          <input type="hidden" id="editSubmissionId">
          <input type="hidden" id="editFilename">

          <div class="field">
            <label>Change Background Color</label>
            <input type="hidden" id="editBgColor" value="">
            <div class="color-swatches">
              <button type="button" class="swatch active" data-color="" title="Keep Original Background">Original</button>
              <button type="button" class="swatch" data-color="#ffffff" style="background:#ffffff; color:#000;" title="Strict White">White</button>
              <button type="button" class="swatch" data-color="#3b82f6" style="background:#3b82f6; color:#fff;" title="Passport Blue">Blue</button>
              <button type="button" class="swatch" data-color="#f3f4f6" style="background:#f3f4f6; color:#000;" title="Off-White">Light Gray</button>
              <button type="button" class="swatch" data-color="#ef4444" style="background:#ef4444; color:#fff;" title="Red">Red</button>
              <label class="color-picker-wrap" title="Pick custom color">
                🎨 Custom
                <input type="color" id="bgColorPicker" value="#ffffff">
              </label>
            </div>
          </div>

          <div class="field">
            <label for="brightnessRange">
              Brightness: <strong id="brightnessValue">0</strong>
            </label>
            <input type="range" id="brightnessRange" min="-100" max="100" value="0" step="1">
            <div class="range-labels">
              <span>-100 (Darker)</span>
              <span>0</span>
              <span>+100 (Brighter)</span>
            </div>
          </div>

          <div class="field">
            <label for="sharpnessRange">
              Sharpness: <strong id="sharpnessValue">1.0</strong>
            </label>
            <input type="range" id="sharpnessRange" min="1.0" max="3.0" value="1.0" step="0.1">
            <div class="range-labels">
              <span>1.0 (Normal)</span>
              <span>2.0</span>
              <span>3.0 (Sharper)</span>
            </div>
          </div>

          <div class="field-row" style="display:flex; gap:12px;">
            <div class="field" style="flex:1;">
              <label for="resWidth">Width (px)</label>
              <input type="number" id="resWidth" placeholder="Width" min="100" max="4000">
            </div>
            <div class="field" style="flex:1;">
              <label for="resHeight">Height (px)</label>
              <input type="number" id="resHeight" placeholder="Height" min="100" max="4000">
            </div>
          </div>

          <button type="button" id="resetEditsBtn" class="btn btn-outline" style="width:100%; margin-top:8px;">
            🔄 Reset Controls
          </button>
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button type="button" id="cancelEditBtn" class="btn btn-outline">Cancel</button>
      <button type="button" id="saveEditBtn" class="btn btn-primary">
        💾 Save & Replace Original
      </button>
    </div>
  </div>
</div>

<div id="toastHost"></div>

<style>
.color-swatches {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 6px;
}
.swatch {
  border: 1px solid var(--navy-600);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  background: var(--navy-700);
  color: var(--ink-100);
  transition: all 0.15s ease;
}
.swatch.active, .swatch:hover {
  outline: 2px solid var(--brand-500);
  outline-offset: 1px;
}
.color-picker-wrap {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--navy-700);
  border: 1px solid var(--navy-600);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.color-picker-wrap input[type="color"] {
  border: none;
  background: none;
  width: 22px;
  height: 22px;
  padding: 0;
  cursor: pointer;
}
.modal-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(6, 10, 23, 0.82);
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 999;
  padding: 20px;
}
.modal-overlay.hidden { display: none; }

.modal-card {
  background: var(--navy-800);
  border: 1px solid var(--navy-600);
  border-radius: var(--radius-lg);
  width: 100%;
  max-width: 960px;
  box-shadow: var(--shadow-lift);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.modal-header {
  padding: 18px 24px;
  border-bottom: 1px solid var(--navy-700);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.modal-header h3 { margin: 0; font-size: 18px; color: #fff; }
.close-btn {
  background: none; border: none; color: var(--ink-500); font-size: 24px; cursor: pointer;
}
.close-btn:hover { color: #fff; }
.modal-body { padding: 24px; }

.edit-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}
@media (max-width: 680px) {
  .edit-grid { grid-template-columns: 1fr; }
}

.preview-box {
  background: var(--navy-900);
  border: 1px solid var(--navy-700);
  border-radius: var(--radius-md);
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.preview-label {
  font-size: 11px;
  color: var(--ink-500);
  margin-bottom: 14px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* Passport photo frame — 7:9 ratio (35×45mm ICAO standard) */
.passport-frame {
  position: relative;
  width: 100%;
  max-width: 320px;
  padding: 18px 28px 18px 18px;
}
.passport-dim {
  position: absolute;
  font-size: 10px;
  font-weight: 600;
  color: var(--ink-500);
  letter-spacing: 0.5px;
  pointer-events: none;
}
.passport-dim-w {
  bottom: 2px;
  left: 50%;
  transform: translateX(-50%);
}
.passport-dim-h {
  right: 0;
  top: 50%;
  transform: translateY(-50%) rotate(90deg);
  transform-origin: center center;
}
.img-container {
  width: 100%;
  aspect-ratio: 7 / 9;
  border-radius: 6px;
  overflow: hidden;
  background: #e8e8e8;
  border: 2px solid var(--navy-600);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow:
    0 1px 3px rgba(0,0,0,0.25),
    inset 0 0 0 1px rgba(0,0,0,0.15);
}
.img-container canvas {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: filter 0.1s ease;
}
.resolution-info {
  margin-top: 14px;
  font-size: 11.5px;
  color: var(--ink-300);
  text-align: center;
}

.controls-box { display: flex; flex-direction: column; gap: 16px; }
.range-labels { display: flex; justify-content: space-between; font-size: 11px; color: var(--ink-500); margin-top: 4px; }
.field input[type="range"] {
  width: 100%;
  accent-color: var(--brand-500);
  cursor: pointer;
}

.modal-footer {
  padding: 16px 24px;
  border-top: 1px solid var(--navy-700);
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  background: var(--navy-900);
}
</style>

<script>
document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('editModal');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const cancelEditBtn = document.getElementById('cancelEditBtn');
  const saveEditBtn = document.getElementById('saveEditBtn');
  const resetEditsBtn = document.getElementById('resetEditsBtn');

  const editorImage = document.getElementById('editorImage');
  const editorCanvas = document.getElementById('editorCanvas');
  const editSubmissionId = document.getElementById('editSubmissionId');
  const editFilename = document.getElementById('editFilename');

  const brightnessRange = document.getElementById('brightnessRange');
  const brightnessValue = document.getElementById('brightnessValue');
  const sharpnessRange = document.getElementById('sharpnessRange');
  const sharpnessValue = document.getElementById('sharpnessValue');
  const resWidth = document.getElementById('resWidth');
  const resHeight = document.getElementById('resHeight');

  let currentNaturalWidth = 0;
  let currentNaturalHeight = 0;

  let cutoutImage = null;
  let cutoutLoading = false;
  let activeCutoutController = null;
  let colorPreviewCache = {};

  const editBgColor = document.getElementById('editBgColor');
  const bgColorPicker = document.getElementById('bgColorPicker');
  const swatches = document.querySelectorAll('.swatch');

  function setBgColor(color) {
    editBgColor.value = color;
    swatches.forEach(s => {
      s.classList.toggle('active', s.dataset.color === color);
    });
    updateLivePreview();
  }

  swatches.forEach(s => {
    s.addEventListener('click', () => {
      setBgColor(s.dataset.color);
    });
  });

  bgColorPicker.addEventListener('input', (e) => {
    setBgColor(e.target.value);
  });

  async function fetchCutout(id) {
    if (!id) return;
    if (activeCutoutController) {
      activeCutoutController.abort();
    }
    activeCutoutController = new AbortController();
    cutoutLoading = true;
    try {
      const resp = await fetch('edit_photo.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: activeCutoutController.signal,
        body: JSON.stringify({
          submission_id: id,
          get_cutout: true
        })
      });
      const resData = await resp.json();
      if (resData.success && resData.cutout_url && editSubmissionId.value == id) {
        const img = new Image();
        img.onload = () => {
          if (editSubmissionId.value == id) {
            cutoutImage = img;
            cutoutLoading = false;
            if (editBgColor.value) {
              updateLivePreview();
            }
          }
        };
        img.src = resData.cutout_url;
      } else {
        cutoutLoading = false;
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        cutoutLoading = false;
      }
    }
  }

  function drawDynamicBackdrop(ctx, w, h, bgHex) {
    const hex = (bgHex || '').replace('#', '');
    const r = parseInt(hex.substring(0, 2) || 'ff', 16);
    const g = parseInt(hex.substring(2, 4) || 'ff', 16);
    const b = parseInt(hex.substring(4, 6) || 'ff', 16);

    const cx = w * 0.5;
    const cy = h * 0.38;
    const radius = Math.max(w, h) * 0.82;

    const grad = ctx.createRadialGradient(cx, cy, radius * 0.05, cx, cy, radius);

    if ((bgHex || '').toLowerCase() === '#ffffff') {
      grad.addColorStop(0, '#ffffff');
      grad.addColorStop(1, '#f1f5f9');
    } else {
      const rInner = Math.min(255, Math.round(r * 1.12));
      const gInner = Math.min(255, Math.round(g * 1.12));
      const bInner = Math.min(255, Math.round(b * 1.12));

      const rOuter = Math.max(0, Math.round(r * 0.88));
      const gOuter = Math.max(0, Math.round(g * 0.88));
      const bOuter = Math.max(0, Math.round(b * 0.88));

      grad.addColorStop(0, `rgb(${rInner}, ${gInner}, ${bInner})`);
      grad.addColorStop(1, `rgb(${rOuter}, ${gOuter}, ${bOuter})`);
    }

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  }

  function updateLivePreview() {
    const b = parseInt(brightnessRange.value, 10);
    const s = parseFloat(sharpnessRange.value);

    if (brightnessValue) brightnessValue.textContent = (b > 0 ? '+' : '') + b;
    if (sharpnessValue) sharpnessValue.textContent = s.toFixed(1);

    if (!editorImage || !editorImage.complete || !editorImage.naturalWidth) return;

    const w = editorImage.naturalWidth;
    const h = editorImage.naturalHeight;
    editorCanvas.width = w;
    editorCanvas.height = h;
    const ctx = editorCanvas.getContext('2d');

    const targetHex = editBgColor.value;
    const cssBrightness = 1 + (b / 100);
    const contrastSim = 1 + ((s - 1) * 0.25);
    editorCanvas.style.filter = `brightness(${cssBrightness}) contrast(${contrastSim})`;

    // Case 1: "Original" background selected
    if (!targetHex || targetHex === '') {
      ctx.drawImage(editorImage, 0, 0, w, h);
      return;
    }

    // Case 2: Subject cutout ready -> Dynamic studio backdrop aligned with subject
    if (cutoutImage && cutoutImage.complete && cutoutImage.naturalWidth) {
      drawDynamicBackdrop(ctx, w, h, targetHex);
      ctx.drawImage(cutoutImage, 0, 0, w, h);
      return;
    }

    // Case 3: Cached preview image exists for targetHex
    if (colorPreviewCache[targetHex] && colorPreviewCache[targetHex].complete) {
      ctx.drawImage(colorPreviewCache[targetHex], 0, 0, w, h);
      return;
    }

    // Case 4: Cutout not yet loaded -> render dynamic backdrop + subject fallback
    drawDynamicBackdrop(ctx, w, h, targetHex);
    ctx.drawImage(editorImage, 0, 0, w, h);

    if (!cutoutLoading && editSubmissionId.value) {
      fetchCutout(editSubmissionId.value);
    }

    // Request server preview as backstop
    if (window._aiPreviewDebounce) clearTimeout(window._aiPreviewDebounce);
    window._aiPreviewDebounce = setTimeout(async () => {
      if (editBgColor.value !== targetHex) return;
      try {
        const resp = await fetch('edit_photo.php', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            submission_id: editSubmissionId.value,
            bg_color: targetHex,
            brightness: b,
            sharpness: s,
            preview: true
          })
        });
        const resData = await resp.json();
        if (resData.success && resData.preview_url && editBgColor.value === targetHex) {
          const aiImg = new Image();
          aiImg.onload = () => {
            colorPreviewCache[targetHex] = aiImg;
            if (editBgColor.value === targetHex && (!cutoutImage || !cutoutImage.complete)) {
              editorCanvas.width = aiImg.naturalWidth;
              editorCanvas.height = aiImg.naturalHeight;
              const aiCtx = editorCanvas.getContext('2d');
              aiCtx.drawImage(aiImg, 0, 0);
            }
          };
          aiImg.src = resData.preview_url;
        }
      } catch (err) {}
    }, 150);
  }

  brightnessRange.addEventListener('input', updateLivePreview);
  sharpnessRange.addEventListener('input', updateLivePreview);

  document.querySelectorAll('.edit-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.id;
      const filename = btn.dataset.filename;
      editSubmissionId.value = id;
      editFilename.value = filename;

      // Abort any pending cutout requests
      if (activeCutoutController) {
        activeCutoutController.abort();
        activeCutoutController = null;
      }

      // Reset controls & state
      brightnessRange.value = 0;
      sharpnessRange.value = 1.0;
      resWidth.value = 1254;
      resHeight.value = 1612;
      cutoutImage = null;
      cutoutLoading = false;
      colorPreviewCache = {};
      editBgColor.value = '';
      swatches.forEach(s => s.classList.toggle('active', s.dataset.color === ''));

      // Clear the canvas immediately so previous image is never shown
      const ctx = editorCanvas.getContext('2d');
      ctx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
      editorCanvas.width = 1;
      editorCanvas.height = 1;

      // Reset source
      editorImage.src = '';
      editorImage.onload = null;

      modal.classList.remove('hidden');

      // Load new selected image cleanly
      const loadImg = new Image();
      loadImg.onload = () => {
        if (editSubmissionId.value !== id) return;
        editorImage.src = loadImg.src;
        currentNaturalWidth = loadImg.naturalWidth;
        currentNaturalHeight = loadImg.naturalHeight;
        updateLivePreview();
      };
      loadImg.src = `/uploads/${filename}?t=${Date.now()}`;
    });
  });

  function closeModal() {
    modal.classList.add('hidden');
    if (activeCutoutController) {
      activeCutoutController.abort();
      activeCutoutController = null;
    }
    const ctx = editorCanvas.getContext('2d');
    ctx.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
    editorImage.src = '';
    editorImage.onload = null;
  }

  closeModalBtn.addEventListener('click', closeModal);
  cancelEditBtn.addEventListener('click', closeModal);

  resetEditsBtn.addEventListener('click', () => {
    brightnessRange.value = 0;
    sharpnessRange.value = 1.0;
    resWidth.value = 1254;
    resHeight.value = 1612;
    setBgColor('');
    updateLivePreview();
  });

  saveEditBtn.addEventListener('click', async () => {
    const id = editSubmissionId.value;
    const filename = editFilename.value;
    const brightness = parseInt(brightnessRange.value, 10);
    const sharpness = parseFloat(sharpnessRange.value);
    const width = parseInt(resWidth.value, 10);
    const height = parseInt(resHeight.value, 10);
    const bgColor = editBgColor.value;

    saveEditBtn.disabled = true;
    saveEditBtn.textContent = 'Saving & Replacing…';

    try {
      const resp = await fetch('edit_photo.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          submission_id: id,
          brightness,
          sharpness,
          width,
          height,
          bg_color: bgColor
        })
      });

      const data = await resp.json();
      if (!resp.ok || data.error) {
        throw new Error(data.error || 'Failed to edit photo.');
      }

      // Update thumbnail image in the table with cache buster
      const thumb = document.getElementById(`thumb-${id}`);
      if (thumb) {
        thumb.src = `/uploads/${filename}?t=${Date.now()}`;
      }

      showToast('Photo edited and replaced as original upload!');
      closeModal();
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      saveEditBtn.disabled = false;
      saveEditBtn.textContent = '💾 Save & Replace Original';
    }
  });
});

function showToast(msg) {
  const host = document.getElementById('toastHost');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  host.appendChild(t);
  setTimeout(() => t.remove(), 3200);
}
</script>

</body>
</html>
