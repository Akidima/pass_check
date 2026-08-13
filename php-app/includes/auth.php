<?php
/**
 * Simple session-based admin authentication.
 */

if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

function admin_login(string $username, string $password): bool {
    $pdo = get_db();
    $stmt = $pdo->prepare("SELECT * FROM admins WHERE username = ?");
    $stmt->execute([$username]);
    $admin = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($admin && password_verify($password, $admin['password_hash'])) {
        $_SESSION['admin_id'] = $admin['id'];
        $_SESSION['admin_username'] = $admin['username'];
        return true;
    }
    return false;
}

function admin_logout(): void {
    unset($_SESSION['admin_id'], $_SESSION['admin_username']);
    session_destroy();
}

function require_admin(): void {
    if (empty($_SESSION['admin_id'])) {
        header('Location: login.php');
        exit;
    }
}

function is_admin_logged_in(): bool {
    return !empty($_SESSION['admin_id']);
}
