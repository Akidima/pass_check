<?php
/**
 * API endpoint: receives an uploaded/captured photo from the frontend,
 * forwards it to the Python verification microservice along with the
 * admin-configured criteria, stores the result, and returns JSON.
 */

// Disable HTML error output for API responses so warnings/notices won't pollute JSON responses
ini_set('display_errors', '0');
error_reporting(E_ALL);

require_once __DIR__ . '/../includes/db.php';

header('Content-Type: application/json');

function json_error(string $message, int $code = 400): void {
    http_response_code($code);
    echo json_encode(['error' => $message]);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    json_error('Method not allowed.', 405);
}

if (empty($_FILES['photo']) || $_FILES['photo']['error'] !== UPLOAD_ERR_OK) {
    json_error('No photo uploaded or upload error occurred.');
}

$allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
$finfo = new finfo(FILEINFO_MIME_TYPE);
$mime = $finfo->file($_FILES['photo']['tmp_name']);

if (!in_array($mime, $allowedTypes, true)) {
    json_error('Invalid file type. Only JPEG, PNG, or WEBP images are allowed.');
}

$maxBytes = 12 * 1024 * 1024;
if ($_FILES['photo']['size'] > $maxBytes) {
    json_error('File too large. Maximum size is 12MB.');
}

$studentName = trim($_POST['student_name'] ?? '');
$studentId = trim($_POST['student_id'] ?? '');
$source = ($_POST['source'] ?? 'upload') === 'camera' ? 'camera' : 'upload';

// Build enabled criteria map from DB (admin's live configuration — never trust client input for this)
$enabledCriteria = get_enabled_criteria();

// Tunable params passed to Python service (e.g. min_pass_criteria setting)
$params = [
    'min_pass_criteria' => (int)get_setting('min_pass_criteria', '4'),
];

// White-background limits are controlled only by the administrator's persisted
// settings. Do not accept client-provided thresholds for this compliance check.
$backgroundParamDefaults = [
    'bg_min_value' => '235',
    'bg_max_saturation' => '18',
    'bg_max_delta_e' => '10',
    'bg_min_white_coverage' => '70',
    'bg_max_nonwhite_component_coverage' => '30',
    'bg_max_luminance_range' => '100',
    'bg_reject_dark_value' => '210',
    'bg_max_dark_coverage' => '5',
    'bg_reject_colored_saturation' => '30',
    'bg_max_colored_coverage' => '5',
    'bg_border_fraction' => '0.12',
];
foreach ($backgroundParamDefaults as $key => $default) {
    $params[$key] = (float)get_setting($key, $default);
}

// Sharpness / blur quality policy (admin-controlled, not client-configurable).
$blurPolicyDefaults = [
    'blur_threshold' => '80',
    'blur_severe_threshold' => '15',
    'blur_soft_fail' => '1',
    'bg_blur_adaptive_tolerance' => '1',
];
foreach ($blurPolicyDefaults as $key => $default) {
    $params[$key] = (float)get_setting($key, $default);
}

$serviceUrl = rtrim(get_setting('python_service_url', 'http://127.0.0.1:5001'), '/') . '/verify';

$cfile = new CURLFile($_FILES['photo']['tmp_name'], $mime, $_FILES['photo']['name']);
$postFields = [
    'photo' => $cfile,
    'criteria' => json_encode($enabledCriteria),
    'params' => json_encode($params),
];

$ch = curl_init($serviceUrl);
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $postFields,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT => 30,
]);

$response = curl_exec($ch);
$curlErrno = curl_errno($ch);
$curlError = curl_error($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
if (PHP_VERSION_ID < 80000) {
    curl_close($ch);
}

if ($curlErrno) {
    json_error('Verification service is unreachable. Please ensure the Python service is running. (' . $curlError . ')', 503);
}

$result = json_decode($response, true);
if ($result === null) {
    json_error('Invalid response from verification service.', 502);
}

if (isset($result['error']) && $httpCode >= 400) {
    json_error($result['error'], $httpCode);
}

$isPassed = !empty($result['overall_passed']);

// Save uploaded photo permanently only if it passed the threshold
$storedName = '';
if ($isPassed) {
    $uploadsDir = __DIR__ . '/../uploads';
    if (!is_dir($uploadsDir)) {
        mkdir($uploadsDir, 0777, true);
    }
    $ext = pathinfo($_FILES['photo']['name'], PATHINFO_EXTENSION) ?: 'jpg';
    $safeExt = preg_replace('/[^a-zA-Z0-9]/', '', $ext);
    $storedName = uniqid('photo_', true) . '.' . $safeExt;
    $targetPath = $uploadsDir . '/' . $storedName;

    // Automatically format background to studio-grade pure white (#ffffff) matching chat format
    $editUrl = rtrim(get_setting('python_service_url', 'http://127.0.0.1:5001'), '/') . '/edit-photo';
    $mimeType = mime_content_type($_FILES['photo']['tmp_name']) ?: 'image/jpeg';
    $cfile = new CURLFile($_FILES['photo']['tmp_name'], $mimeType, basename($_FILES['photo']['tmp_name']));
    $postFieldsBg = [
        'photo' => $cfile,
        'bg_color' => '#ffffff'
    ];

    $chBg = curl_init($editUrl);
    curl_setopt_array($chBg, [
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $postFieldsBg,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 20,
    ]);
    $formattedBytes = curl_exec($chBg);
    $httpCodeBg = curl_getinfo($chBg, CURLINFO_HTTP_CODE);
    if (PHP_VERSION_ID < 80000) {
        curl_close($chBg);
    }

    if ($httpCodeBg === 200 && !empty($formattedBytes)) {
        file_put_contents($targetPath, $formattedBytes);
    } else {
        move_uploaded_file($_FILES['photo']['tmp_name'], $targetPath);
    }
} else {
    // Delete temporary uploaded file so failed photo is not stored/available
    if (file_exists($_FILES['photo']['tmp_name'])) {
        @unlink($_FILES['photo']['tmp_name']);
    }
}

// Persist submission record
$pdo = get_db();
$stmt = $pdo->prepare("INSERT INTO submissions (student_name, student_id, filename, source, overall_passed, result_json)
                        VALUES (?, ?, ?, ?, ?, ?)");
$stmt->execute([
    $studentName,
    $studentId,
    $storedName,
    $source,
    $isPassed ? 1 : 0,
    json_encode($result),
]);

echo json_encode($result);
