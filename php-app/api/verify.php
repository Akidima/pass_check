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

// Forward background verification settings
$verificationSettings = get_verification_settings();
$params['background_strictness'] = $verificationSettings['strictness'];
$params['background_near_white_acceptance'] = $verificationSettings['near_white_acceptance'];

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

// Shared secret required by the Python service (Authorization: Bearer).
// Read from the environment only — never hardcoded and never sent to the browser.
$serviceToken = getenv('PORTAL_SHARED_SECRET');
if ($serviceToken === false || $serviceToken === '') {
    json_error('Verification service is not configured (missing PORTAL_SHARED_SECRET).', 503);
}
// Timeout budget. Measured warm single-request verification is well over 10
// seconds because tie detection runs a Faster R-CNN on CPU, and requests queue
// behind the sync Gunicorn workers under concurrent uploads. The previous 30s
// verify timeout was below the observed queued latency, so legitimate requests
// were reported to applicants as "service unreachable" while Python kept
// working on them.
//
// Invariant: VERIFY_TIMEOUT + EDIT_TIMEOUT must stay below the Gunicorn
// `timeout` setting, so Gunicorn never kills a worker that PHP is still
// waiting on.
const VERIFY_TIMEOUT_SECONDS = 60;
const EDIT_TIMEOUT_SECONDS = 30;

/**
 * Absolute instant (unix milliseconds) at which this caller stops waiting.
 * The Python service drops queued work whose deadline has already passed
 * instead of spending CPU on a response nobody will read.
 */
function deadline_header(int $timeoutSeconds): string {
    return 'X-Request-Deadline: ' . (string)(int)round((microtime(true) + $timeoutSeconds) * 1000);
}

$serviceHeaders = ['Authorization: Bearer ' . $serviceToken];

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
    CURLOPT_HTTPHEADER => array_merge($serviceHeaders, [deadline_header(VERIFY_TIMEOUT_SECONDS)]),
    CURLOPT_CONNECTTIMEOUT => 5,
    CURLOPT_TIMEOUT => VERIFY_TIMEOUT_SECONDS,
]);

$response = curl_exec($ch);
$curlErrno = curl_errno($ch);
$curlError = curl_error($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
if (PHP_VERSION_ID < 80000) {
    curl_close($ch);
}

if ($curlErrno) {
    // Log the transport detail server-side with a correlation id; never echo
    // cURL text (which can contain internal hostnames, ports and paths) to the
    // applicant's browser.
    $correlationId = bin2hex(random_bytes(8));
    error_log(sprintf(
        '[verify %s] cURL error %d calling verification service: %s',
        $correlationId, $curlErrno, $curlError
    ));
    // 28 == CURLE_OPERATION_TIMEDOUT. Spelled numerically because PHP exposes
    // the constant under two different names across versions.
    $isTimeout = ($curlErrno === 28);
    json_error(
        $isTimeout
            ? 'The verification service is busy and did not respond in time. Please try again in a moment. (Ref: ' . $correlationId . ')'
            : 'The verification service is temporarily unavailable. Please try again shortly. (Ref: ' . $correlationId . ')',
        503
    );
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
        mkdir($uploadsDir, 0750, true);
    }
    // The output from /edit-photo is always JPEG. Never derive a stored
    // extension from the applicant-controlled original filename.
    $storedName = bin2hex(random_bytes(16)) . '.jpg';
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
        CURLOPT_HTTPHEADER => array_merge($serviceHeaders, [deadline_header(EDIT_TIMEOUT_SECONDS)]),
        CURLOPT_CONNECTTIMEOUT => 5,
        CURLOPT_TIMEOUT => EDIT_TIMEOUT_SECONDS,
    ]);
    $formattedBytes = curl_exec($chBg);
    $httpCodeBg = curl_getinfo($chBg, CURLINFO_HTTP_CODE);
    if (PHP_VERSION_ID < 80000) {
        curl_close($chBg);
    }

    if ($httpCodeBg === 200 && !empty($formattedBytes)) {
        file_put_contents($targetPath, $formattedBytes);
        $result['processing'] = ['status' => 'completed', 'format' => 'jpeg'];
    } else {
        // Editing failed, so retain the verified original with an extension
        // derived from the server-detected MIME type, not the user filename.
        $fallbackExt = [
            'image/jpeg' => 'jpg',
            'image/png' => 'png',
            'image/webp' => 'webp',
        ][$mime] ?? 'jpg';
        $storedName = bin2hex(random_bytes(16)) . '.' . $fallbackExt;
        $targetPath = $uploadsDir . '/' . $storedName;
        move_uploaded_file($_FILES['photo']['tmp_name'], $targetPath);
        // The original photo was accepted, but the standardised white-background
        // JPEG was NOT produced. Record that distinctly so nothing downstream
        // assumes the stored file is the generated output.
        error_log(sprintf(
            '[verify] /edit-photo failed with HTTP %d; stored unstandardised original %s',
            $httpCodeBg, $storedName
        ));
        $result['processing'] = [
            'status' => 'failed',
            'format' => $fallbackExt,
            'detail' => 'Original accepted, but white-background generation did not run.',
        ];
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
