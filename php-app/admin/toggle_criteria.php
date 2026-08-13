<?php
ini_set('display_errors', '0');
error_reporting(E_ALL);

require_once __DIR__ . '/../includes/db.php';
require_once __DIR__ . '/../includes/auth.php';

header('Content-Type: application/json');

if (!is_admin_logged_in()) {
    http_response_code(401);
    echo json_encode(['error' => 'Unauthorized']);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

$input = json_decode(file_get_contents('php://input'), true);
$key = $input['criteria_key'] ?? '';
$enabled = !empty($input['enabled']) ? 1 : 0;

if (!$key) {
    http_response_code(400);
    echo json_encode(['error' => 'Missing criteria_key']);
    exit;
}

$pdo = get_db();
$stmt = $pdo->prepare("UPDATE criteria SET enabled = ? WHERE criteria_key = ?");
$stmt->execute([$enabled, $key]);

echo json_encode(['success' => true, 'criteria_key' => $key, 'enabled' => (bool)$enabled]);
