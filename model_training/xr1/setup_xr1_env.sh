#!/usr/bin/env bash
# Create the XR-1 environment from env/ lock files; set XR1_ENV to its absolute path afterwards.
# Usage: bash xr1/setup_xr1_env.sh <new environment folder>
# Installs CUDA build tools, Python packages, flash-attention and Qwen processor files.
# Requires conda, a C++ compiler and network access. DeepSpeed builds FusedAdam on first use.
# XR-1 itself remains a separate clone supplied through PYTHONPATH. Attribution: NOTICE.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${1:?usage: setup_xr1_env.sh <env folder>}"
[ ! -e "$ENV" ] || { echo "$ENV exists; delete it or choose another folder" >&2; exit 1; }
FLASH_ATTN_WHEEL='flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
FLASH_ATTN_URL='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
FLASH_ATTN_SHA256=75c51f34bb93c5a4438d6b767547ca6c22a9bfc6b29b5646dda98e943cd75d99

conda create -y -q --override-channels -c conda-forge -p "$ENV" --file "$HERE/env/conda-explicit.txt"
"$ENV/bin/pip" install -q --no-deps -r "$HERE/env/requirements.lock.txt"
DOWNLOAD="$(mktemp -d)"
curl -fsSL -o "$DOWNLOAD/$FLASH_ATTN_WHEEL" "$FLASH_ATTN_URL"
echo "$FLASH_ATTN_SHA256  $DOWNLOAD/$FLASH_ATTN_WHEEL" | sha256sum -c -
"$ENV/bin/pip" install -q --no-deps "$DOWNLOAD/$FLASH_ATTN_WHEEL"
rm -rf "$DOWNLOAD"

# torch's CUDA headers include cusparse.h and others; with CUDA_HOME = this env they must be in its
# include folder for the FusedAdam build (they come with the pip nvidia-* packages)
for package in cublas cufft curand cusolver cusparse; do
  for header in "$ENV"/lib/python3.10/site-packages/nvidia/$package/include/*; do
    target="$ENV/targets/x86_64-linux/include/$(basename "$header")"
    [ -e "$target" ] || ln -s "$header" "$target"
  done
done

"$ENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-VL-4B-Instruct", allow_patterns=["*.json", "*.txt", "*.jinja"]))
PY
"$ENV/bin/python" -c "import torch, flash_attn, deepspeed, decord, lightning, transformers; \
print('torch', torch.__version__, 'deepspeed', deepspeed.__version__, 'transformers', transformers.__version__)"
echo "done: set XR1_ENV=\"$(realpath "$ENV")\" in config.sh"
