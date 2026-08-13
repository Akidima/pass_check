<?php
require_once __DIR__ . '/includes/db.php';
$pageTitle = get_setting('university_name', 'University Passport Photo Verification System') . ' — Student Portal';
$criteria = get_all_criteria();
$enabledCriteria = array_values(array_filter($criteria, fn($c) => (int)$c['enabled'] === 1));
include __DIR__ . '/includes/header.php';
?>

<div class="topbar">
  <div class="brand">
    <div class="mark">🪪</div>
    <div>
      <?= htmlspecialchars(get_setting('university_name', 'University Passport System')) ?>
      <span class="sub">Student Photo Verification</span>
    </div>
  </div>
  <div class="nav-links">
    <a href="index.php" class="pill-link active">Student Portal</a>
    <a href="admin/login.php" class="pill-link">Admin</a>
  </div>
</div>

<div class="shell">

  <div class="hero">
    <div class="eyebrow">✨ AI-Powered Verification</div>
    <h1>Get your official passport photo<br>verified in seconds</h1>
    <p>Upload a photo or use your camera. Our system automatically checks it against your university's official passport photo requirements — instantly.</p>
  </div>

  <div class="grid-2">
    <!-- LEFT: Capture / Upload -->
    <div class="card panel">
      <h3 class="section-title">Submit Your Photo</h3>
      <p class="section-desc">Choose upload or camera capture below.</p>

      <div class="mode-switch">
        <button class="mode-btn active" id="modeUploadBtn" type="button">📁 Upload Photo</button>
        <button class="mode-btn" id="modeCameraBtn" type="button">📷 Use Camera</button>
      </div>

      <!-- Upload mode -->
      <div id="uploadPane">
        <div class="dropzone" id="dropzone">
          <div class="icon">🖼️</div>
          <div class="primary">Click to browse or drag & drop</div>
          <div class="secondary">JPEG, PNG or WEBP — up to 12MB</div>
        </div>
        <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp" class="hidden">
        <img id="uploadPreview" class="preview-thumb hidden" alt="Preview">
      </div>

      <!-- Camera mode -->
      <div id="cameraPane" class="hidden">
        <div class="camera-wrap">
          <video id="cameraVideo" autoplay playsinline muted></video>
          <img id="cameraCaptured" class="captured hidden" alt="Captured photo">
          <div class="camera-guide" id="cameraGuide"></div>
        </div>
        <div class="camera-controls">
          <button class="btn btn-outline" id="startCameraBtn" type="button">Start Camera</button>
          <button class="btn btn-primary hidden" id="snapBtn" type="button">📸 Capture</button>
          <button class="btn btn-outline hidden" id="retakeBtn" type="button">↺ Retake</button>
        </div>
      </div>

      <canvas id="captureCanvas" class="hidden"></canvas>

      <div class="field" style="margin-top:22px;">
        <label for="studentName">Full Name</label>
        <input type="text" id="studentName" placeholder="e.g. Adaeze Okafor">
      </div>
      <div class="field">
        <label for="studentId">Student ID</label>
        <input type="text" id="studentId" placeholder="e.g. 2021/CS/0142">
      </div>

      <button class="btn btn-primary btn-full" id="verifyBtn" type="button" disabled>
        <span id="verifyBtnLabel">Verify Photo</span>
      </button>
      <div id="serviceWarning" class="alert alert-error hidden" style="margin-top:14px;"></div>
    </div>

    <!-- RIGHT: Live criteria + results -->
    <div class="card panel">
      <h3 class="section-title">Verification Checklist</h3>
      <p class="section-desc">Your photo is checked against these <?= count($enabledCriteria) ?> active requirements.</p>

      <div id="resultBanner"></div>

      <div class="criteria-list" id="criteriaList">
        <?php foreach ($enabledCriteria as $c): ?>
          <div class="criteria-item pending" data-key="<?= htmlspecialchars($c['criteria_key']) ?>">
            <div class="status-dot">•</div>
            <div class="body">
              <strong><?= htmlspecialchars($c['label']) ?></strong>
              <span><?= htmlspecialchars($c['description']) ?></span>
            </div>
          </div>
        <?php endforeach; ?>
        <?php if (empty($enabledCriteria)): ?>
          <p style="color:var(--ink-500); font-size:13px;">No criteria are currently enabled by the admin. Any photo will pass.</p>
        <?php endif; ?>
      </div>
    </div>
  </div>

  <div class="steps">
    <div class="card step">
      <div class="num">1</div>
      <h4>Capture or Upload</h4>
      <p>Take a photo with your camera or upload an existing one.</p>
    </div>
    <div class="card step">
      <div class="num">2</div>
      <h4>AI Verification</h4>
      <p>Our computer-vision engine checks background, pose, glasses, resolution & more.</p>
    </div>
    <div class="card step">
      <div class="num">3</div>
      <h4>Instant Result</h4>
      <p>Pass instantly, or get a clear reason and retry immediately.</p>
    </div>
  </div>

  <div class="footer-note">
    Powered by a hybrid PHP + Python (OpenCV / MediaPipe) verification pipeline · For support, contact your university IT helpdesk.
  </div>
</div>

<script src="/assets/js/app.js"></script>
</body>
</html>
