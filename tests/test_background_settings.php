<?php
/**
 * Persistence + propagation tests for the two-control background-validation
 * configuration (run with: php tests/test_background_settings.php).
 *
 * Exercises the real SQLite settings table through db.php:
 *   - defaults seed correctly (strictness + near-white acceptance)
 *   - set_setting -> get_verification_settings round-trips both controls
 *   - corrupted values fall back to safe defaults
 *   - the legacy Advanced-Threshold keys are migrated away
 *   - an explicit legacy near-white choice carries over to the new key
 *
 * The verification API forwards exactly these values to the Python service,
 * whose presets map them to the image-analysis thresholds (covered by the
 * Python suite in python-service/tests/test_background_config_flow.py).
 */

// Isolated throwaway database representing a FRESH installation, so the
// defaults and migration behaviour are tested without touching the real
// developer/administrator database.
// IMPORTANT: the env override must be set BEFORE requiring db.php, because
// DB_PATH is defined (from the environment) at include time.
$tmpDb = tempnam(sys_get_temp_dir(), 'pass_check_test_') . '.sqlite';
putenv('PASS_CHECK_DB_PATH=' . $tmpDb);

require_once __DIR__ . '/../php-app/includes/db.php';

$failures = 0;
$checks = 0;

function check(bool $condition, string $label): void {
    global $failures, $checks;
    $checks++;
    if ($condition) {
        echo "  ok  {$label}\n";
    } else {
        $failures++;
        echo "FAIL  {$label}\n";
    }
}

echo "== defaults ==\n";
$settings = get_verification_settings();
check($settings['strictness'] === 'standard', 'default strictness is standard');
check($settings['near_white_acceptance'] === 'auto', 'default near-white acceptance is auto');

echo "\n== persistence round-trip ==\n";
foreach (['strict', 'standard', 'relaxed', 'accept_all'] as $level) {
    set_setting('background_strictness', $level);
    check(get_verification_settings()['strictness'] === $level, "strictness '{$level}' persists and reads back");
}
foreach (['auto', '1', '0'] as $acceptance) {
    set_setting('background_near_white_acceptance', $acceptance);
    check(get_verification_settings()['near_white_acceptance'] === $acceptance, "near-white acceptance '{$acceptance}' persists and reads back");
}

echo "\n== corrupted values fall back safely ==\n";
set_setting('background_strictness', 'DROP TABLE admins; --');
check(get_verification_settings()['strictness'] === 'standard', 'injected strictness value falls back to standard');
set_setting('background_near_white_acceptance', 'maybe');
check(get_verification_settings()['near_white_acceptance'] === 'auto', 'invalid acceptance value falls back to auto');

echo "\n== legacy Advanced-Threshold migration ==\n";
// Migration runs against its own FRESH database so it is independent of
// the writes made by the round-trip checks above.
$migrationDb = tempnam(sys_get_temp_dir(), 'pass_check_mig_') . '.sqlite';
putenv('PASS_CHECK_DB_PATH=' . $migrationDb);
// Simulate an upgraded installation carrying the removed per-field rows.
$pdo = get_db();
$insertLegacy = $pdo->prepare("INSERT OR REPLACE INTO settings (setting_key, setting_value) VALUES (?, ?)");
foreach ([
    'bg_min_value' => '235',
    'bg_max_saturation' => '18',
    'bg_max_delta_e' => '10',
    'bg_near_white_min_l_star' => '93',
    'bg_near_white_max_chroma' => '10',
    'bg_near_white_max_b_star' => '9',
    'bg_min_white_coverage' => '30',
    'bg_max_nonwhite_component_coverage' => '30',
    'bg_max_luminance_range' => '100',
    'bg_reject_dark_value' => '210',
    'bg_max_dark_coverage' => '5',
    'bg_reject_colored_saturation' => '30',
    'bg_max_colored_coverage' => '5',
    'bg_border_fraction' => '0.12',
] as $key => $value) {
    $insertLegacy->execute([$key, $value]);
}
// init_schema() runs on every get_db(); call it again to run the migration.
init_schema($pdo);

$stmt = $pdo->query("SELECT COUNT(*) FROM settings WHERE setting_key LIKE 'bg\\_%' ESCAPE '\\' AND setting_key NOT IN ('bg_blur_adaptive_tolerance')");
$remainingLegacyFields = (int)$stmt->fetchColumn();
check($remainingLegacyFields === 0, "all 14 per-field bg_* threshold rows are deleted by migration");

// An explicit legacy near-white choice must survive into the new key.
$insertLegacy->execute(['background_near_white_acceptance', 'auto']);
$insertLegacy->execute(['bg_near_white_enabled', '1']);
init_schema($pdo);
check(get_verification_settings()['near_white_acceptance'] === '1', "legacy forced-on switch migrates to the new tri-state key");
check(!$pdo->query("SELECT COUNT(*) FROM settings WHERE setting_key = 'bg_near_white_enabled'")->fetchColumn(), "legacy switch key is removed after migration");

@unlink($tmpDb);
@unlink($migrationDb);
echo "\n{$checks} checks, {$failures} failures\n";
exit($failures === 0 ? 0 : 1);
