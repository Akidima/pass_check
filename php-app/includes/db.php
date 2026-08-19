<?php
/**
 * Database bootstrap (SQLite via PDO).
 * Creates schema on first run.
 */

define('DB_PATH', __DIR__ . '/../data/database.sqlite');

function get_db(): PDO {
    static $pdo = null;
    if ($pdo !== null) {
        return $pdo;
    }

    $dataDir = dirname(DB_PATH);
    if (!is_dir($dataDir)) {
        mkdir($dataDir, 0777, true);
    }

    $pdo = new PDO('sqlite:' . DB_PATH);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    $pdo->exec('PRAGMA foreign_keys = ON;');

    init_schema($pdo);
    return $pdo;
}

function init_schema(PDO $pdo): void {
    $pdo->exec("
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    ");

    $pdo->exec("
        CREATE TABLE IF NOT EXISTS criteria (
            criteria_key TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            description TEXT,
            enabled INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER DEFAULT 0
        );
    ");

    $pdo->exec("
        CREATE TABLE IF NOT EXISTS settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT
        );
    ");

    $pdo->exec("
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            student_id TEXT,
            filename TEXT NOT NULL,
            source TEXT DEFAULT 'upload',
            overall_passed INTEGER NOT NULL,
            result_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    ");

    // Seed default admin (username: admin / password: Admin@123) if none exists
    $count = (int)$pdo->query("SELECT COUNT(*) FROM admins")->fetchColumn();
    if ($count === 0) {
        $stmt = $pdo->prepare("INSERT INTO admins (username, password_hash) VALUES (?, ?)");
        $stmt->execute(['admin', password_hash('Admin@123', PASSWORD_DEFAULT)]);
    }

    // Seed default criteria
    $defaults = [
        ['single_face', 'Single Face Detection', 'Exactly one person must be visible in the photo.', 1, 1],
        ['face_framing', 'Face Size & Centering', 'Face must be properly sized and centered in the frame.', 1, 2],
        ['no_glasses', 'No Eyeglasses', 'Eyeglasses are not permitted in the photo.', 0, 3],
        ['white_background', 'Strictly White Background', 'Background must be plain and white.', 1, 4],
        ['require_tie', 'Tie / Formal Neckwear Required', 'A tie or formal neckwear must be worn.', 0, 5],
        ['no_tie', 'No Necktie', 'Visible neckties are not permitted in the photo (learned detector).', 0, 12],
        ['min_resolution', 'Minimum Resolution', 'Photo must meet minimum resolution requirements.', 1, 6],
        ['passport_ratio', 'Passport Size Aspect Ratio', 'Photo must match standard passport aspect ratio.', 0, 7],
        ['no_blur', 'Sharpness (No Blur)', 'Photo must be sharp and in focus.', 1, 8],
        ['brightness', 'Proper Lighting / Exposure', 'Photo must be properly lit, not too dark or bright.', 1, 9],
        ['head_pose', 'Straight Head Pose', 'Head must face forward without excessive tilt or turn.', 1, 10],
        ['eyes_open', 'Eyes Open', 'Eyes must be open and visible.', 1, 11],
    ];

    $check = $pdo->prepare("SELECT COUNT(*) FROM criteria WHERE criteria_key = ?");
    $insert = $pdo->prepare("INSERT INTO criteria (criteria_key, label, description, enabled, sort_order) VALUES (?, ?, ?, ?, ?)");
    foreach ($defaults as $d) {
        $check->execute([$d[0]]);
        if ((int)$check->fetchColumn() === 0) {
            $insert->execute($d);
        }
    }

    // Seed default settings (Python service URL, thresholds)
    $settingDefaults = [
        'python_service_url' => 'http://127.0.0.1:5001',
        'university_name' => 'University Passport Photo Verification System',
        'max_attempts' => '5',
        'min_pass_criteria' => '4',
        // White-background verification settings. These are forwarded to the
        // Python CV service for every request and are the administrator's
        // source of truth (the default requires 70% visible white background).
        'bg_min_value' => '235',
        'bg_max_saturation' => '18',
        'bg_max_delta_e' => '10',
        'bg_min_white_coverage' => '30',
        'bg_max_nonwhite_component_coverage' => '30',
        'bg_max_luminance_range' => '100',
        'bg_reject_dark_value' => '210',
        'bg_max_dark_coverage' => '5',
        'bg_reject_colored_saturation' => '30',
        'bg_max_colored_coverage' => '5',
        'bg_border_fraction' => '0.12',
        // Sharpness / blur quality policy. These control the tiered blur
        // severity system that prevents moderately blurred but genuinely
        // white-background photos from being rejected unnecessarily.
        'blur_threshold' => '80',
        'blur_severe_threshold' => '15',
        'blur_soft_fail' => '1',
        'bg_blur_adaptive_tolerance' => '1',
    ];
    $checkS = $pdo->prepare("SELECT COUNT(*) FROM settings WHERE setting_key = ?");
    $insertS = $pdo->prepare("INSERT INTO settings (setting_key, setting_value) VALUES (?, ?)");
    foreach ($settingDefaults as $k => $v) {
        $checkS->execute([$k]);
        if ((int)$checkS->fetchColumn() === 0) {
            $insertS->execute([$k, $v]);
        }
    }

    // Upgrade installations that still have the old generated defaults. Values
    // explicitly changed by an administrator are left untouched.
    $legacyDefaults = [
        'bg_max_nonwhite_component_coverage' => ['0', '30'],
        'bg_max_luminance_range' => ['3', '100'],
        'bg_min_value' => ['250', '235'],
        'bg_max_saturation' => ['6', '18'],
        'bg_max_delta_e' => ['3', '10'],
        'bg_reject_dark_value' => ['220', '210'],
        'bg_max_dark_coverage' => ['0', '5'],
        'bg_reject_colored_saturation' => ['20', '30'],
        'bg_max_colored_coverage' => ['0', '5'],
    ];
    $readSetting = $pdo->prepare("SELECT setting_value FROM settings WHERE setting_key = ?");
    $updateSetting = $pdo->prepare("UPDATE settings SET setting_value = ? WHERE setting_key = ?");
    foreach ($legacyDefaults as $key => [$legacyValue, $replacement]) {
        $readSetting->execute([$key]);
        if ($readSetting->fetchColumn() === $legacyValue) {
            $updateSetting->execute([$replacement, $key]);
        }
    }
}

function get_setting(string $key, string $default = ''): string {
    $pdo = get_db();
    $stmt = $pdo->prepare("SELECT setting_value FROM settings WHERE setting_key = ?");
    $stmt->execute([$key]);
    $val = $stmt->fetchColumn();
    return $val !== false ? $val : $default;
}

function set_setting(string $key, string $value): void {
    $pdo = get_db();
    $stmt = $pdo->prepare("INSERT INTO settings (setting_key, setting_value) VALUES (?, ?)
                            ON CONFLICT(setting_key) DO UPDATE SET setting_value = excluded.setting_value");
    $stmt->execute([$key, $value]);
}

function get_enabled_criteria(): array {
    $pdo = get_db();
    $rows = $pdo->query("SELECT criteria_key, enabled FROM criteria")->fetchAll(PDO::FETCH_ASSOC);
    $map = [];
    foreach ($rows as $r) {
        $map[$r['criteria_key']] = (bool)((int)$r['enabled']);
    }
    return $map;
}

function get_all_criteria(): array {
    $pdo = get_db();
    return $pdo->query("SELECT * FROM criteria ORDER BY sort_order ASC")->fetchAll(PDO::FETCH_ASSOC);
}
