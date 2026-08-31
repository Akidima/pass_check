"""Download and verify the model weights this service needs, before deployment.

Run this once on the target host as part of deployment, BEFORE routing any
student traffic:

    python scripts/prestage_models.py

Why this exists: torchvision and rembg fetch their weights from the internet on
first use. Without pre-staging, the first student upload either pays a ~350 MB
download or fails outright on a firewalled host, and the download happens inside
a request that PHP is timing out on.

Set TORCH_HOME and U2NET_HOME first so the weights land in the service account's
own read-only model directory rather than a developer's home cache.

Exits non-zero if any required artifact is missing, so deployment can gate on it.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# torchvision publishes this filename and hash for
# FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.
TORCHVISION_TIE_CHECKPOINT = "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report(label: str, path: Path) -> bool:
    if not path.exists():
        print(f"  MISSING  {label}: {path}")
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  ok       {label}: {path} ({size_mb:.1f} MB)")
    print(f"           sha256={_sha256(path)}")
    return True


def stage_tie_detector() -> bool:
    """Load the configured tie backend, which downloads its weights if absent."""
    print("Tie detector")
    backend = os.environ.get("TIE_DETECTOR_BACKEND", "auto").strip().lower()
    if backend == "auto":
        print(
            "  WARNING  TIE_DETECTOR_BACKEND=auto silently falls back to the "
            "generic COCO detector. Set it explicitly for deployment."
        )
    try:
        from tie_detector import get_tie_detector

        detector = get_tie_detector()
    except Exception as exc:
        print(f"  FAILED   could not build the tie detector: {type(exc).__name__}: {exc}")
        return False
    print(f"  ok       backend loaded, version={getattr(detector, 'version', 'unknown')}")

    torch_home = Path(
        os.environ.get("TORCH_HOME") or (Path.home() / ".cache" / "torch")
    )
    checkpoint = torch_home / "hub" / "checkpoints" / TORCHVISION_TIE_CHECKPOINT
    if backend in {"auto", "coco"}:
        return _report("COCO checkpoint", checkpoint)
    return True


def stage_rembg() -> bool:
    """Create the configured rembg session, which downloads its ONNX model if absent."""
    model = os.environ.get("REMBG_MODEL", "u2netp").strip().lower() or "u2netp"
    print(f"Background removal (rembg {model})")
    try:
        from rembg import new_session

        new_session(model)
    except Exception as exc:
        print(f"  FAILED   could not create the rembg session: {type(exc).__name__}: {exc}")
        return False
    model_home = Path(os.environ.get("U2NET_HOME") or (Path.home() / ".u2net"))
    filename = {
        "u2netp": "u2netp.onnx",
        "u2net": "u2net.onnx",
        "u2net_human_seg": "u2net_human_seg.onnx",
        "silueta": "silueta.onnx",
        "isnet-general-use": "isnet-general-use.onnx",
    }.get(model, f"{model}.onnx")
    return _report(f"{model} model", model_home / filename)


def stage_mediapipe() -> bool:
    """Prove MediaPipe can actually start a graph on this host.

    MediaPipe ships its own assets, but it needs a working runtime context. A
    host where the graph cannot start serves requests while rejecting every
    photograph, so failing here is far better than failing in production.
    """
    print("MediaPipe face detection / landmarks")
    try:
        import numpy as np

        import verify

        blank = np.full((480, 360, 3), 240, dtype=np.uint8)
        verify._detect_faces(blank)
        verify._face_mesh_landmarks(blank)
    except Exception as exc:
        print(f"  FAILED   MediaPipe raised {type(exc).__name__}: {exc}")
        return False

    health = verify.perception_health()
    if not health["landmarks_available"]:
        print(f"  FAILED   landmarks unusable on this host: {health['mediapipe_last_error']}")
        print("           head_pose and eyes_open would reject every applicant.")
        return False
    print("  ok       graph starts and landmarks are available")
    return True


def main() -> int:
    # Import from the service directory regardless of the caller's cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("Pre-staging model artifacts")
    print(f"  TORCH_HOME={os.environ.get('TORCH_HOME', '(default ~/.cache/torch)')}")
    print(f"  U2NET_HOME={os.environ.get('U2NET_HOME', '(default ~/.u2net)')}")
    print()

    results = {
        "tie detector": stage_tie_detector(),
        "rembg": stage_rembg(),
        "mediapipe": stage_mediapipe(),
    }
    print()

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"INCOMPLETE: {', '.join(failed)}. Do not route traffic to this host yet.")
        return 1
    print("All model artifacts are staged and usable. Verify /ready before routing traffic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
