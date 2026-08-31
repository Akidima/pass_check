<?php
/**
 * Admin API endpoint: edits a submission photo (brightness, resolution, sharpness)
 * and replaces the original image file in uploads/ directory.
 */

ini_set('display_errors', '0');
error_reporting(E_ALL);

require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';
require_admin();

header('Content-Type: application/json');

function json_response(array $data, int $code = 200): void {
    http_response_code($code);
    echo json_encode($data);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    json_response(['error' => 'Method not allowed.'], 405);
}

$rawInput = file_get_contents('php://input');
$input = json_decode($rawInput, true);

if (!is_array($input)) {
    $input = $_POST;
}

$id = (int)($input['submission_id'] ?? $input['id'] ?? 0);
if ($id <= 0) {
    json_response(['error' => 'Invalid submission ID.'], 400);
}

$pdo = get_db();
$stmt = $pdo->prepare("SELECT * FROM submissions WHERE id = ?");
$stmt->execute([$id]);
$submission = $stmt->fetch(PDO::FETCH_ASSOC);

if (!$submission || empty($submission['filename'])) {
    json_response(['error' => 'Submission or photo file not found.'], 404);
}

$filename = $submission['filename'];
$filePath = __DIR__ . '/../uploads/' . $filename;

if (!file_exists($filePath)) {
    json_response(['error' => 'Original photo file does not exist on server.'], 404);
}

$brightness = (float)($input['brightness'] ?? 0);
$width = isset($input['width']) && (int)$input['width'] > 0 ? (int)$input['width'] : null;
$height = isset($input['height']) && (int)$input['height'] > 0 ? (int)$input['height'] : null;
$sharpness = (float)($input['sharpness'] ?? 1.0);
$bgColor = trim($input['bg_color'] ?? '');

$serviceUrl = rtrim(get_setting('python_service_url', 'http://127.0.0.1:5001'), '/') . '/edit-photo';
$serviceToken = getenv('PORTAL_SHARED_SECRET');
if ($serviceToken === false || $serviceToken === '') {
    json_response(['error' => 'Verification service is not configured (missing PORTAL_SHARED_SECRET).'], 503);
}

$mime = mime_content_type($filePath) ?: 'image/jpeg';
$cfile = new CURLFile($filePath, $mime, basename($filePath));

$isCutout = !empty($input['get_cutout']);
$isPreview = !empty($input['preview']);
$wantsBackground = $bgColor !== '';

$postFields = [
    'photo' => $cfile,
    'brightness' => (string)$brightness,
    'sharpness' => (string)$sharpness,
];

if ($isCutout) {
    $postFields['get_cutout'] = '1';
}

if ($wantsBackground) {
    $postFields['bg_color'] = $bgColor;
}

if ($width !== null && $height !== null) {
    $postFields['width'] = (string)$width;
    $postFields['height'] = (string)$height;
}

$ch = curl_init($serviceUrl);
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_POSTFIELDS => $postFields,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['Authorization: Bearer ' . $serviceToken],
    CURLOPT_CONNECTTIMEOUT => 5,
    CURLOPT_TIMEOUT => 60,
]);

$editedBytes = curl_exec($ch);
$curlErrno = curl_errno($ch);
$httpCode = (int)curl_getinfo($ch, CURLINFO_HTTP_CODE);
if (PHP_VERSION_ID < 80000) {
    curl_close($ch);
}

if (!$curlErrno && $httpCode === 200 && !empty($editedBytes)) {
    if ($isCutout) {
        json_response([
            'success' => true,
            'cutout_url' => 'data:image/png;base64,' . base64_encode($editedBytes),
            'filename' => $filename,
            'id' => $id,
        ]);
    }
    if ($isPreview) {
        json_response([
            'success' => true,
            'preview_url' => 'data:image/jpeg;base64,' . base64_encode($editedBytes),
            'filename' => $filename,
            'id' => $id,
        ]);
    }
    file_put_contents($filePath, $editedBytes);
    json_response([
        'success' => true,
        'message' => 'Photo edited and replaced successfully.',
        'filename' => $filename,
        'id' => $id,
        'timestamp' => time()
    ]);
}

$pythonError = null;
if (is_string($editedBytes) && $editedBytes !== '') {
    $decoded = json_decode($editedBytes, true);
    if (is_array($decoded) && !empty($decoded['error'])) {
        $pythonError = (string)$decoded['error'];
    }
}

// GD cannot isolate a subject. Do not pretend a background change succeeded.
if ($isCutout || $wantsBackground) {
    json_response([
        'error' => $pythonError
            ?: ('Background replacement failed (HTTP ' . ($httpCode ?: 0) . '). Check that the Python service is running and PORTAL_SHARED_SECRET matches.'),
    ], 502);
}

// Fallback to PHP GD if Python service is unreachable
if (extension_loaded('gd')) {
    $imgData = @file_get_contents($filePath);
    $gdImg = @imagecreatefromstring($imgData);
    if ($gdImg !== false) {
        $origW = imagesx($gdImg);
        $origH = imagesy($gdImg);

        // 1. Resize if width and height specified
        if ($width && $height && ($width !== $origW || $height !== $origH)) {
            $resized = imagecreatetruecolor($width, $height);
            imagecopyresampled($resized, $gdImg, 0, 0, 0, 0, $width, $height, $origW, $origH);
            imagedestroy($gdImg);
            $gdImg = $resized;
        }

        // 2. Brightness adjustment
        if ($brightness != 0) {
            $b_offset = (int)max(-255, min(255, $brightness * 2.55));
            imagefilter($gdImg, IMG_FILTER_BRIGHTNESS, $b_offset);
        }

        // 3. Sharpness adjustment via convolution matrix
        if ($sharpness > 1.0) {
            $sharpFactor = min(3.0, $sharpness - 1.0);
            $matrix = [
                [-1 * $sharpFactor, -1 * $sharpFactor, -1 * $sharpFactor],
                [-1 * $sharpFactor, 9 * $sharpFactor + 1, -1 * $sharpFactor],
                [-1 * $sharpFactor, -1 * $sharpFactor, -1 * $sharpFactor]
            ];
            $divisor = array_sum(array_map('array_sum', $matrix));
            imageconvolution($gdImg, $matrix, $divisor, 0);
        }

        imagejpeg($gdImg, $filePath, 95);
        imagedestroy($gdImg);

        json_response([
            'success' => true,
            'message' => 'Photo edited and replaced successfully (via GD).',
            'filename' => $filename,
            'id' => $id,
            'timestamp' => time()
        ]);
    }
}

json_response(['error' => 'Failed to process photo edit. Python microservice or GD library unavailable.'], 500);
