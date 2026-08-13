<?php
require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';

if (is_admin_logged_in()) {
    header('Location: dashboard.php');
    exit;
}

$error = '';
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim($_POST['username'] ?? '');
    $password = $_POST['password'] ?? '';
    if (admin_login($username, $password)) {
        header('Location: dashboard.php');
        exit;
    }
    $error = 'Invalid username or password.';
}

$pageTitle = 'Admin Login';
include __DIR__ . '/../includes/header.php';
?>

<div class="login-wrap">
  <div class="card login-card">
    <div class="brand">
      <div class="mark">🛡️</div>
      <div>Admin Portal</div>
    </div>
    <p class="lead">Manage passport photo verification criteria</p>

    <?php if ($error): ?>
      <div class="alert alert-error"><?= htmlspecialchars($error) ?></div>
    <?php endif; ?>

    <form method="POST">
      <div class="field">
        <label for="username">Username</label>
        <input type="text" id="username" name="username" placeholder="admin" required autofocus>
      </div>
      <div class="field">
        <label for="password">Password</label>
        <input type="password" id="password" name="password" placeholder="••••••••" required>
      </div>
      <button type="submit" class="btn btn-primary btn-full">Sign In</button>
    </form>

    <div class="hint-box">
      Default credentials: <strong>admin</strong> / <strong>Admin@123</strong><br>
      Please change this password after first login.
    </div>

    <p style="text-align:center; margin-top:20px;">
      <a href="../index.php" class="pill-link" style="display:inline-block;">← Back to Student Portal</a>
    </p>
  </div>
</div>

</body>
</html>
