<?php
require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';
require_admin();

$message = '';
$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['action']) && $_POST['action'] === 'update_settings') {
        set_setting('university_name', trim($_POST['university_name'] ?? 'University Passport Photo Verification System'));
        set_setting('python_service_url', trim($_POST['python_service_url'] ?? 'http://127.0.0.1:5001'));
        $minCriteria = max(1, (int)($_POST['min_pass_criteria'] ?? 4));
        set_setting('min_pass_criteria', (string)$minCriteria);
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
      <p class="section-desc">Configure the display name, verification URL, and minimum pass criteria threshold.</p>
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
