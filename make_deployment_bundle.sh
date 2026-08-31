#!/usr/bin/env bash
#
# Build a self-contained deployment bundle for the Python verification service.
#
# Produces dist/passport-verification-service.tar.gz containing everything the
# target host needs. The target does not need this repository, git, or network
# access to any private host.
#
# Usage:
#   bash make_deployment_bundle.sh                 # code only (~400 KB)
#   bash make_deployment_bundle.sh --with-weights  # also embed model weights (~335 MB)
#
# Use --with-weights when the target host is firewalled or offline. It copies
# the already-downloaded torchvision and rembg weights from this machine's
# caches so the target never downloads anything.

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/python-service"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dist"
NAME="passport-verification-service"
STAGE="$OUT/$NAME"

WITH_WEIGHTS=0
[[ "${1:-}" == "--with-weights" ]] && WITH_WEIGHTS=1

rm -rf "$OUT"
mkdir -p "$STAGE"/{scripts,tests,benchmarks,models}

# --- Runtime modules. This is the closed first-party import graph: app.py ->
# service_auth.py + verify.py + training_capture.py; verify.py ->
# tie_visibility.py + tie_detector.py. Omitting any one of them breaks startup.
cp "$SRC"/app.py \
   "$SRC"/service_auth.py \
   "$SRC"/verify.py \
   "$SRC"/tie_detector.py \
   "$SRC"/tie_visibility.py \
   "$SRC"/training_capture.py \
   "$SRC"/gunicorn.conf.py \
   "$STAGE"/

# --- Dependencies, configuration template and the deployment guide.
cp "$SRC"/requirements.txt \
   "$SRC"/requirements-dev.txt \
   "$SRC"/.env.example \
   "$SRC"/DEPLOY.md \
   "$STAGE"/

# --- Deploy-time tooling: stages and verifies model artifacts, exits non-zero
# if anything is unusable, so deployment can gate on it.
cp "$SRC"/scripts/prestage_models.py "$STAGE"/scripts/

# --- Acceptance suite. The deployment gate is "no failing tests on the target
# host", so the tests have to travel with the code.
cp "$SRC"/tests/*.py "$STAGE"/tests/

# --- Tie-detection validation tools. Latency is hardware-dependent and the
# input-resolution default is justified by measured decision equivalence, so
# the target host needs to be able to re-verify both.
cp "$SRC"/benchmarks/benchmark_tie_detector.py \
   "$SRC"/benchmarks/verify_tie_resolution_equivalence.py \
   "$STAGE"/benchmarks/

# --- Policy template, only used if a custom checkpoint is deployed later.
cp "$SRC"/models/tie_detector.policy.example.json "$STAGE"/models/

# --- Optional: embed model weights for an offline target.
if [[ "$WITH_WEIGHTS" == "1" ]]; then
  TORCH_CKPT="${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
  U2NET_MODEL="${U2NET_HOME:-$HOME/.u2net}/u2net.onnx"
  missing=0
  for f in "$TORCH_CKPT" "$U2NET_MODEL"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: expected weight file not found: $f" >&2
      missing=1
    fi
  done
  if [[ "$missing" == "1" ]]; then
    echo "Run 'python scripts/prestage_models.py' on THIS machine first." >&2
    exit 1
  fi
  mkdir -p "$STAGE"/model-weights/torch/hub/checkpoints "$STAGE"/model-weights/u2net
  cp "$TORCH_CKPT" "$STAGE"/model-weights/torch/hub/checkpoints/
  cp "$U2NET_MODEL" "$STAGE"/model-weights/u2net/
  cat > "$STAGE"/model-weights/README.txt <<'EOF'
Pre-downloaded model weights, for a target host without internet access.

Copy them to the directories named by TORCH_HOME and U2NET_HOME in your .env,
preserving this layout:

    torch/hub/checkpoints/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
        -> $TORCH_HOME/hub/checkpoints/
    u2net/u2net.onnx
        -> $U2NET_HOME/

Then run scripts/prestage_models.py to verify them. It prints each file's
SHA-256 and exits non-zero if anything is unusable. Expected checksums:

    fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth
        dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf
    u2net.onnx
        8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491
EOF
fi

# --- Nothing generated or environment-specific should travel.
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true

tar -czf "$OUT/$NAME.tar.gz" -C "$OUT" "$NAME"

echo "Bundle contents:"
find "$STAGE" -type f | sed "s|$STAGE|  .|" | sort
echo
echo "Staged directory: $STAGE"
echo "Archive:          $OUT/$NAME.tar.gz"
du -sh "$OUT/$NAME.tar.gz" | awk '{print "Archive size:     " $1}'
echo
echo "On the target host:"
echo "  tar -xzf $NAME.tar.gz && cd $NAME && cat DEPLOY.md"
