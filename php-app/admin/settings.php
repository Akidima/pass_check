<?php
require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';
require_admin();

$message = '';
$error = '';
$whiteBackgroundFields = [
    'bg_min_value' => ['Minimum RGB value', '235', 0, 255, 1, 'Every sampled background pixel must be at least this bright. Higher values are stricter (default 235 for mobile cameras).'],
    'bg_max_saturation' => ['Maximum HSV saturation', '18', 0, 255, 1, 'Rejects tinted backgrounds while permitting warm/cool white lighting (default 18). Lower values are stricter.'],
    'bg_max_delta_e' => ['Maximum LAB distance to pure white', '10', 0, 150, 0.1, 'Perceptual colour-distance limit from pure white (default 10.0 for real-world lighting). Lower values are stricter.'],
    'bg_min_white_coverage' => ['Minimum white-background coverage (%)', '30', 0, 100, 0.01, 'At least this percentage of visible background must meet all white limits.'],
    'bg_max_nonwhite_component_coverage' => ['Maximum contiguous non-white region (%)', '30', 0, 100, 0.01, 'Limits the size of a single non-white background patch.'],
    'bg_max_luminance_range' => ['Maximum luminance range (L*)', '100', 0, 100, 0.1, 'Limits shadows and gradients across the sampled background. Lower values are stricter.'],
    'bg_reject_dark_value' => ['Dark/black rejection value', '210', 0, 255, 1, 'Pixels darker than this are treated as black/dark background. Higher values are stricter.'],
    'bg_max_dark_coverage' => ['Maximum dark/black coverage (%)', '5', 0, 100, 0.01, 'Maximum allowed dark/black background coverage (default 5% allows minor hair/bezel artifacts).'],
    'bg_reject_colored_saturation' => ['Colour rejection saturation', '30', 0, 255, 1, 'Pixels above this HSV saturation are treated as coloured background. Lower values are stricter.'],
    'bg_max_colored_coverage' => ['Maximum coloured coverage (%)', '5', 0, 100, 0.01, 'Maximum allowed coloured background coverage (default 5% allows minor edge artifacts).'],
    'bg_border_fraction' => ['Border region to inspect (fraction)', '0.12', 0.03, 0.30, 0.01, 'Outer image fraction sampled as background; the detected subject area is excluded.'],
];

$blurPolicyFields = [
    'blur_threshold' => ['Ideal sharpness threshold (Laplacian variance)', '80', 1, 500, 1, 'Images with sharpness above this value pass unconditionally. Lower values accept blurrier photos.'],
    'blur_severe_threshold' => ['Severe blur threshold', '15', 1, 200, 1, 'Images below this sharpness are rejected as too blurred for reliable verification.'],
    'blur_soft_fail' => ['Enable soft-fail mode', '1', 0, 1, 1, 'When enabled (1), moderately blurred photos can still pass if the white background check passes. Set to 0 for strict blur enforcement.'],
    'bg_blur_adaptive_tolerance' => ['Adaptive background tolerance for blur', '1', 0, 1, 1, 'When enabled (1), slightly relaxes background brightness/ΔE limits for blurred images to compensate for optical color spreading.'],
];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['action']) && $_POST['action'] === 'update_settings') {
        set_setting('university_name', trim($_POST['university_name'] ?? 'University Passport Photo Verification System'));
        set_setting('python_service_url', trim($_POST['python_service_url'] ?? 'http://127.0.0.1:5001'));
        $minCriteria = max(1, (int)($_POST['min_pass_criteria'] ?? 4));
        set_setting('min_pass_criteria', (string)$minCriteria);
        foreach ($whiteBackgroundFields as $key => [, $default, $min, $max]) {
            $value = filter_var($_POST[$key] ?? null, FILTER_VALIDATE_FLOAT);
            if ($value === false) {
                $value = (float)$default;
            }
            $value = min((float)$max, max((float)$min, (float)$value));
            set_setting($key, (string)$value);
        }
        foreach ($blurPolicyFields as $key => [, $default, $min, $max]) {
            $value = filter_var($_POST[$key] ?? null, FILTER_VALIDATE_FLOAT);
            if ($value === false) {
                $value = (float)$default;
            }
            $value = min((float)$max, max((float)$min, (float)$value));
            set_setting($key, (string)$value);
        }
        $message = 'Settings updated successfully.';
    }

    if (isset($_POST['action']) && $_POST['action'] === 'change_password') {
        $current = $_POST['current_password'] ?? '';
        $new = $_POST['new_password'] ?? '';
        $confirm = $_POST['confirm_password'] ?? '';

        $pdo = get_db();
        $stmt = $pdo->prepare("SELECT * FROM admins WHERE id = ?");
        $stmt->execute([$_SESSION['admin_id']]);
        $admin = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$admin || !password_verify($current, $admin['password_hash'])) {
            $error = 'Current password is incorrect.';
        } elseif (strlen($new) < 8) {
            $error = 'New password must be at least 8 characters.';
        } elseif ($new !== $confirm) {
            $error = 'New password and confirmation do not match.';
        } else {
            $upd = $pdo->prepare("UPDATE admins SET password_hash = ? WHERE id = ?");
            $upd->execute([password_hash($new, PASSWORD_DEFAULT), $_SESSION['admin_id']]);
            $message = 'Password changed successfully.';
        }
    }
}

$uniName = get_setting('university_name');
$serviceUrl = get_setting('python_service_url');
$minPassCriteria = get_setting('min_pass_criteria', '4');
$whiteBackgroundValues = [];
foreach ($whiteBackgroundFields as $key => [, $default]) {
    $whiteBackgroundValues[$key] = get_setting($key, $default);
}
$blurPolicyValues = [];
foreach ($blurPolicyFields as $key => [, $default]) {
    $blurPolicyValues[$key] = get_setting($key, $default);
}

$pageTitle = 'Settings';
include __DIR__ . '/../includes/header.php';
?>

<div class="topbar">
  <div class="brand">
    <div class="mark">🛡️</div>
    <div>Settings <span class="sub">System configuration</span></div>
  </div>
  <div class="nav-links">
    <a href="dashboard.php" class="pill-link">Dashboard</a>
    <a href="submissions.php" class="pill-link">Submissions</a>
    <a href="settings.php" class="pill-link active">Settings</a>
    <a href="logout.php" class="pill-link" style="color:#ff9d9d;">Logout</a>
  </div>
</div>

<div class="shell">
  <?php if ($message): ?><div class="alert alert-success"><?= htmlspecialchars($message) ?></div><?php endif; ?>
  <?php if ($error): ?><div class="alert alert-error"><?= htmlspecialchars($error) ?></div><?php endif; ?>

  <div class="grid-2">
    <div class="card panel">
      <h3 class="section-title">System Settings</h3>
      <p class="section-desc">Configure the display name, verification URL, minimum pass criteria threshold, and strict white-background limits.</p>
      <form method="POST">
        <input type="hidden" name="action" value="update_settings">
        <div class="field">
          <label>University / Institution Name</label>
          <input type="text" name="university_name" value="<?= htmlspecialchars($uniName) ?>" required>
        </div>
        <div class="field">
          <label>Python Verification Service URL</label>
          <input type="url" name="python_service_url" value="<?= htmlspecialchars($serviceUrl) ?>" required>
        </div>
        <div class="field">
          <label>Minimum Passing Criteria Threshold</label>
          <input type="number" name="min_pass_criteria" value="<?= htmlspecialchars($minPassCriteria) ?>" min="1" max="11" required>
          <span style="font-size:12px; color:var(--ink-500); display:block; margin-top:4px;">
            If 4 or more criteria are enabled by admin, a photo passes if it satisfies at least this many criteria (default: 4).
          </span>
        </div>
        <h4 class="section-title" style="margin-top:28px; font-size:16px;">Strict White Background</h4>
        <p class="section-desc">These values are sent to the verification service for every request. The default requires at least 70% visible white background and rejects visible dark, black, or coloured background.</p>
        <?php foreach ($whiteBackgroundFields as $key => [$label, $default, $min, $max, $step, $help]): ?>
          <div class="field">
            <label><?= htmlspecialchars($label) ?></label>
            <input type="number" name="<?= htmlspecialchars($key) ?>" value="<?= htmlspecialchars($whiteBackgroundValues[$key]) ?>"
                   min="<?= htmlspecialchars((string)$min) ?>" max="<?= htmlspecialchars((string)$max) ?>" step="<?= htmlspecialchars((string)$step) ?>" required>
            <span style="font-size:12px; color:var(--ink-500); display:block; margin-top:4px;"><?= htmlspecialchars($help) ?></span>
          </div>
        <?php endforeach; ?>
        <h4 class="section-title" style="margin-top:28px; font-size:16px;">Sharpness / Blur Quality Policy</h4>
        <p class="section-desc">Controls how sharpness (blur) is evaluated relative to background whiteness. When soft-fail is enabled, moderately blurred photos with a confirmed white background can still pass instead of being rejected.</p>
        <?php foreach ($blurPolicyFields as $key => [$label, $default, $min, $max, $step, $help]): ?>
          <div class="field">
            <label><?= htmlspecialchars($label) ?></label>
            <input type="number" name="<?= htmlspecialchars($key) ?>" value="<?= htmlspecialchars($blurPolicyValues[$key]) ?>"
                   min="<?= htmlspecialchars((string)$min) ?>" max="<?= htmlspecialchars((string)$max) ?>" step="<?= htmlspecialchars((string)$step) ?>" required>
            <span style="font-size:12px; color:var(--ink-500); display:block; margin-top:4px;"><?= htmlspecialchars($help) ?></span>
          </div>
        <?php endforeach; ?>
        <button type="submit" class="btn btn-primary btn-full">Save Settings</button>
      </form>
    </div>

    <div class="card panel">
      <h3 class="section-title">Change Password</h3>
      <p class="section-desc">Update your admin login credentials.</p>
      <form method="POST">
        <input type="hidden" name="action" value="change_password">
        <div class="field">
          <label>Current Password</label>
          <input type="password" name="current_password" required>
        </div>
        <div class="field">
          <label>New Password</label>
          <input type="password" name="new_password" required minlength="8">
        </div>
        <div class="field">
          <label>Confirm New Password</label>
          <input type="password" name="confirm_password" required minlength="8">
        </div>
        <button type="submit" class="btn btn-outline btn-full">Update Password</button>
      </form>
    </div>
  </div>
</div>

</body>
</html>
