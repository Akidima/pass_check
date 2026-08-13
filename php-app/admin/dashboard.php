<?php
require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';
require_admin();

$pdo = get_db();
$criteria = get_all_criteria();

$totalSubmissions = (int)$pdo->query("SELECT COUNT(*) FROM submissions")->fetchColumn();
$passedSubmissions = (int)$pdo->query("SELECT COUNT(*) FROM submissions WHERE overall_passed = 1")->fetchColumn();
$failedSubmissions = $totalSubmissions - $passedSubmissions;
$passRate = $totalSubmissions > 0 ? round(($passedSubmissions / $totalSubmissions) * 100) : 0;

$serviceUrl = get_setting('python_service_url', 'http://127.0.0.1:5001');
$uniName = get_setting('university_name', 'University Passport Photo Verification System');

// Check Python service health
$serviceOnline = false;
$ch = curl_init(rtrim($serviceUrl, '/') . '/health');
curl_setopt_array($ch, [CURLOPT_RETURNTRANSFER => true, CURLOPT_TIMEOUT => 3]);
$resp = curl_exec($ch);
if ($resp !== false && curl_getinfo($ch, CURLINFO_HTTP_CODE) === 200) {
    $serviceOnline = true;
}
if (PHP_VERSION_ID < 80000) {
    curl_close($ch);
}

$pageTitle = 'Admin Dashboard';
include __DIR__ . '/../includes/header.php';
?>

<div class="topbar">
  <div class="brand">
    <div class="mark">🛡️</div>
    <div>
      Admin Dashboard
      <span class="sub"><?= htmlspecialchars($uniName) ?></span>
    </div>
  </div>
  <div class="nav-links">
    <a href="../index.php" class="pill-link">View Student Portal</a>
    <a href="submissions.php" class="pill-link">Submissions</a>
    <a href="settings.php" class="pill-link">Settings</a>
    <a href="logout.php" class="pill-link" style="color:#ff9d9d;">Logout</a>
  </div>
</div>

<div class="shell">

  <?php if (!$serviceOnline): ?>
    <div class="alert alert-error" style="margin-bottom:24px;">
      ⚠️ The Python verification service is currently <strong>offline</strong> (<?= htmlspecialchars($serviceUrl) ?>).
      Students will not be able to verify photos until it is started. Run <code>python app.py</code> in the <code>python-service</code> folder.
    </div>
  <?php else: ?>
    <div class="alert alert-success" style="margin-bottom:24px;">
      ✅ Verification service is online at <?= htmlspecialchars($serviceUrl) ?>
    </div>
  <?php endif; ?>

  <div class="stats-row">
    <div class="card stat-card">
      <div class="value"><?= $totalSubmissions ?></div>
      <div class="label">Total Submissions</div>
    </div>
    <div class="card stat-card">
      <div class="value" style="color:var(--ok-500);"><?= $passedSubmissions ?></div>
      <div class="label">Passed</div>
    </div>
    <div class="card stat-card">
      <div class="value" style="color:var(--err-500);"><?= $failedSubmissions ?></div>
      <div class="label">Failed</div>
    </div>
    <div class="card stat-card">
      <div class="value"><?= $passRate ?>%</div>
      <div class="label">Pass Rate</div>
    </div>
  </div>

  <div class="card panel">
    <h3 class="section-title">Verification Criteria</h3>
    <p class="section-desc">Toggle which checks are enforced when students submit photos. Changes apply instantly.</p>

    <div class="criteria-admin-list" id="criteriaAdminList">
      <?php foreach ($criteria as $c): ?>
        <div class="criteria-admin-item">
          <div class="info">
            <strong><?= htmlspecialchars($c['label']) ?></strong>
            <span><?= htmlspecialchars($c['description']) ?></span>
          </div>
          <label class="switch">
            <input type="checkbox" class="criteria-toggle" data-key="<?= htmlspecialchars($c['criteria_key']) ?>"
                   <?= (int)$c['enabled'] === 1 ? 'checked' : '' ?>>
            <span class="track"></span>
          </label>
        </div>
      <?php endforeach; ?>
    </div>
  </div>

  <div class="footer-note">
    Logged in as <strong><?= htmlspecialchars($_SESSION['admin_username']) ?></strong> · Passport Photo Verification System
  </div>
</div>

<div id="toastHost"></div>

<script>
document.querySelectorAll('.criteria-toggle').forEach((el) => {
  el.addEventListener('change', async () => {
    const key = el.dataset.key;
    const enabled = el.checked;
    try {
      const resp = await fetch('toggle_criteria.php', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ criteria_key: key, enabled })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || 'Failed to update');
      showToast(`"${el.closest('.criteria-admin-item').querySelector('strong').textContent}" ${enabled ? 'enabled' : 'disabled'}.`);
    } catch (err) {
      el.checked = !enabled;
      showToast('Error: ' + err.message);
    }
  });
});

function showToast(msg) {
  const host = document.getElementById('toastHost');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  host.appendChild(t);
  setTimeout(() => t.remove(), 2800);
}
</script>

</body>
</html>
