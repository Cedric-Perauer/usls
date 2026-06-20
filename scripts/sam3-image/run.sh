#!/usr/bin/env bash
# Convenience wrapper for running the SAM3 image scripts with GPU onnxruntime.
#
# onnxruntime-gpu loads its CUDA/cuDNN libraries from the NVIDIA pip wheels that
# `uv sync` installs into the venv, but those directories are not on the dynamic
# loader path by default. This wrapper adds them, then forwards to `uv run`.
#
# Usage:
#   ./run.sh inference_v2.py --image ../../assets/kids.jpg --text "shoe" \
#       --model-dir ./onnx-models-v2 --tokenizer /path/to/tokenizer.json
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NV="$DIR/.venv/lib/python3.12/site-packages/nvidia"

if [ -d "$NV" ]; then
  for sub in cu13 cudnn cusparselt nccl nvshmem; do
    [ -d "$NV/$sub/lib" ] && LD_LIBRARY_PATH="$NV/$sub/lib:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
fi

# TensorRT libraries (for onnxruntime's TensorrtExecutionProvider), if installed.
TRT="$DIR/.venv/lib/python3.12/site-packages/tensorrt_libs"
if [ -d "$TRT" ]; then
  export LD_LIBRARY_PATH="$TRT:${LD_LIBRARY_PATH:-}"
  # The pip TensorRT wheels omit libnvinfer_vc_plugin.so.10, but onnxruntime's
  # ONNX->TRT parser dlopen()s it during init and fails fatally if absent. The
  # version-compatibility plugin isn't needed to build normal engines, so a tiny
  # empty stub satisfies the load and lets parsing/engine-build proceed.
  if [ ! -e "$TRT/libnvinfer_vc_plugin.so.10" ] && command -v cc >/dev/null 2>&1; then
    echo 'void _stub(void){}' | cc -shared -fPIC -x c - -o "$TRT/libnvinfer_vc_plugin.so.10" 2>/dev/null || true
  fi
fi

exec uv run --project "$DIR" python "$@"
