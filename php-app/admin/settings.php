<?php
require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';
require_admin();

$message = '';
$error = '';

// Available background strictness levels
$backgroundStrictnessLevels = [
    'strict' => [
        'label' => 'Strict',
        'desc' => 'Pure studio white only. International passport / NYSC standard.',
    ],
    'standard' => [
        'label' => 'Standard',
        'desc' => 'Recommended. Light-grey walls, warm/cool room lighting and mild shadows are accepted.',
    ],
    'relaxed' => [
        'label' => 'Relaxed',
        'desc' => 'Wider tolerance for home-taken photos with imperfect lighting.',
    ],
    'accept_all' => [
        'label' => 'Accept All',
        'desc' => 'Any background except dark or strongly coloured ones. Portal-avatar use only.',
    ],
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

        // Save background strictness level
        $strictnessInput = $_POST['background_strictness'] ?? 'standard';
        $strictness = array_key_exists($strictnessInput, $backgroundStrictnessLevels)
            ? $strictnessInput
            : 'standard';
        set_setting('background_strictness', $strictness);

        // Save near-white acceptance toggle
        $nearWhite = isset($_POST['background_near_white_acceptance']) ? '1' : '0';
        set_setting('background_near_white_acceptance', $nearWhite);

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

// Single authoritative read of the two background-validation controls.
$verificationSettings = get_verification_settings();
$backgroundStrictness = $verificationSettings['strictness'];
$nearWhiteAcceptance = $verificationSettings['near_white_acceptance'];

// The toggle displays the EFFECTIVE state: while nothing has been saved yet
// the stored value is 'auto', meaning "follow the strictness level"
// (off on Strict, on everywhere else). Saving the form pins an explicit
// ON/OFF choice.
$nearWhiteEffective = $nearWhiteAcceptance === 'auto'
    ? ($backgroundStrictness !== 'strict')
    : ($nearWhiteAcceptance === '1');

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
      <p class="section-desc">Configure the display name, verification URL, minimum pass criteria threshold, and white-background validation.</p>
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

        <h4 class="section-title" style="margin-top:28px; font-size:16px;">White Background Validation</h4>
        <p class="section-desc">Two controls define how the background of a submitted photo is judged. The strictness level sets how tolerant the image analysis is; the near-white toggle decides whether slightly tinted off-white backgrounds are acceptable.</p>

        <div class="field">
          <label>White Background Strictness Level</label>
          <div style="display:flex; flex-direction:column; gap:8px; margin-top:4px;">
            <?php foreach ($backgroundStrictnessLevels as $levelKey => $level): ?>
              <label style="display:flex; align-items:flex-start; gap:10px; padding:10px 12px; border:1px solid <?= $backgroundStrictness === $levelKey ? 'var(--accent, #7aa2ff)' : 'var(--navy-600)' ?>; border-radius:var(--radius-sm); cursor:pointer;<?= $backgroundStrictness === $levelKey ? ' background:rgba(122,162,255,0.08);' : '' ?>">
                <input type="radio" name="background_strictness" value="<?= htmlspecialchars($levelKey) ?>"<?= $backgroundStrictness === $levelKey ? ' checked' : '' ?> required style="margin-top:2px;">
                <span>
                  <strong style="font-size:13px;"><?= htmlspecialchars($level['label']) ?></strong>
                  <span style="display:block; font-size:12px; color:var(--ink-500); margin-top:2px;"><?= htmlspecialchars($level['desc']) ?></span>
                </span>
              </label>
            <?php endforeach; ?>
          </div>
          <span style="font-size:12px; color:var(--ink-500); display:block; margin-top:6px;">
            The level automatically configures every internal threshold used by the image analysis engine — brightness, colour-neutrality, coverage and shadow limits. Dark or strongly coloured backgrounds are rejected at every level.
          </span>
        </div>

        <div class="field" style="margin-top:16px;">
          <label style="display:flex; align-items:center; gap:10px; cursor:pointer;">
            <input type="checkbox" name="background_near_white_acceptance" value="1"<?= $nearWhiteEffective ? ' checked' : '' ?> style="width:18px; height:18px;">
            <span>
              <strong style="font-size:13px;">Near-White Background Acceptance</strong>
              <span style="display:block; font-size:12px; color:var(--ink-500); margin-top:2px;">
                ON — bright, almost colourless backgrounds photographed under real-world lighting (slightly dimmed or warm/cool-tinted white walls, light-grey studio backdrops) are accepted. OFF — only pure white qualifies.<?= $nearWhiteAcceptance === 'auto' ? ' Currently following the strictness level default.' : '' ?>
              </span>
            </span>
          </label>
          <span style="font-size:12px; color:var(--ink-500); display:block; margin-top:6px;">
            Beige, cream, blue, green and dark backgrounds are always rejected regardless of this toggle.
          </span>
        </div>

        <details style="margin-top:16px; border:1px solid var(--navy-600); border-radius:var(--radius-sm); padding:0;">
          <summary style="cursor:pointer; padding:12px 16px; font-size:13px; font-weight:600; color:var(--ink-300); user-select:none; list-style:none; display:flex; align-items:center; gap:8px;">
            <span style="display:inline-block; transition:transform 0.2s; font-size:10px;" class="adv-arrow">▶</span>
            Sharpness / Blur Quality Policy
          </summary>
          <div style="padding:4px 16px 16px; border-top:1px solid var(--navy-700);">
            <p style="font-size:12px; color:var(--ink-500); margin:8px 0 14px;">Controls how sharpness (blur) is evaluated relative to background whiteness. When soft-fail is enabled, moderately blurred photos with a confirmed white background can still pass instead of being rejected.</p>
            <?php foreach ($blurPolicyFields as $key => [$label, $default, $min, $max, $step, $help]): ?>
              <div class="field">
                <label><?= htmlspecialchars($label) ?></label>
                <input type="number" name="<?= htmlspecialchars($key) ?>" value="<?= htmlspecialchars($blurPolicyValues[$key]) ?>"
                       min="<?= htmlspecialchars((string)$min) ?>" max="<?= htmlspecialchars((string)$max) ?>" step="<?= htmlspecialchars((string)$step) ?>" required>
                <span style="font-size:12px; color:var(--ink-500); display:block; margin-top:4px;"><?= htmlspecialchars($help) ?></span>
              </div>
            <?php endforeach; ?>
          </div>
        </details>
        <style>
          details[open] .adv-arrow { transform: rotate(90deg); }
        </style>
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
