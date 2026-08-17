"""Reproducible local CPU benchmark for white-background inference.

Measures latency on representative mobile camera resolutions under realistic
conditions (clean white, sensor noise, blur, and compression).
"""

import json
import pathlib
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verify import check_white_background, check_blur, _measure_sharpness  # noqa: E402


SIZES = [(480, 640), (720, 1280), (1080, 1920), (3024, 4032)]
RUNS = 5


def benchmark_white_bg(width, height):
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    params = {}
    # Warm OpenCV kernels before measuring
    check_white_background(image, [], params)
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        result = check_white_background(image, [], params)
        samples.append((time.perf_counter() - started) * 1000.0)
    assert result["passed"], result
    return {
        "check": "white_background_clean",
        "resolution": f"{width}x{height}",
        "runs": RUNS,
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 3),
    }


def benchmark_blur_tiered(width, height):
    """Benchmark the tiered blur check."""
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    for y in range(0, height, 4):
        image[y, :, :] = 0

    check_blur(image, [], {})
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        result = check_blur(image, [], {})
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "check": "blur_tiered",
        "resolution": f"{width}x{height}",
        "severity": result["meta"]["severity"],
        "sharpness": round(result["meta"]["sharpness"], 1),
        "runs": RUNS,
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 3),
    }


def benchmark_blurred_mobile_image(width, height):
    """Benchmark with a Gaussian-blurred and noise-added mobile image."""
    image = np.full((height, width, 3), 242, dtype=np.uint8)
    # Add face rectangle
    fx, fy, fw, fh = int(width * 0.25), int(height * 0.2), int(width * 0.5), int(height * 0.4)
    cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (120, 100, 80), -1)
    image = cv2.GaussianBlur(image, (15, 15), 5)
    params = {}

    check_white_background(image, [{"x": fx, "y": fy, "w": fw, "h": fh}], params)
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        result = check_white_background(image, [{"x": fx, "y": fy, "w": fw, "h": fh}], params)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "check": "white_bg_with_mobile_blur",
        "resolution": f"{width}x{height}",
        "passed": result["passed"],
        "runs": RUNS,
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 3),
    }


if __name__ == "__main__":
    results = []
    print("=== White Background Clean Benchmark ===")
    for w, h in SIZES:
        res = benchmark_white_bg(w, h)
        results.append(res)
        print(f"  {res['resolution']}: {res['median_ms']} ms (median), {res['p95_ms']} ms (p95)")

    print("\n=== Tiered Blur Check Benchmark ===")
    for w, h in SIZES:
        res = benchmark_blur_tiered(w, h)
        results.append(res)
        print(f"  {res['resolution']}: {res['median_ms']} ms (median)")

    print("\n=== Blurred Mobile Image Benchmark ===")
    for w, h in SIZES:
        res = benchmark_blurred_mobile_image(w, h)
        results.append(res)
        print(f"  {res['resolution']}: {res['median_ms']} ms (median), passed={res['passed']}")

    print("\n=== Full JSON Report ===")
    print(json.dumps(results, indent=2))

