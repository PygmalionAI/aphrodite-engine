#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Install optional KV connectors against the active PyTorch/CUDA stack.
# Usage: tools/install_kv_connectors.sh [REQUIREMENTS_DIR]
set -euo pipefail

requirements_dir="${1:-$(dirname "${BASH_SOURCE[0]}")/../requirements}"
uv_target=()
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    uv_target=(--system)
fi

if python3 -c "import torch; raise SystemExit(0 if torch.version.hip is not None else 1)"; then
    uv pip install "${uv_target[@]}" -r "${KV_CONNECTORS_REQUIREMENTS:-${requirements_dir}/kv_connectors_rocm.txt}"
    exit 0
fi

# Resolve the backend before optional packages can change the installed Torch.
CUDA_MAJOR=$(python3 -c 'import torch; print(torch.version.cuda.split(".", 1)[0])')
default_requirements="${requirements_dir}/kv_connectors.txt"
if [[ "${CUDA_MAJOR}" == "12" ]]; then
    default_requirements="${requirements_dir}/kv_connectors_cu12.txt"
fi
requirements_file="${KV_CONNECTORS_REQUIREMENTS:-${default_requirements}}"

constraints_dir=$(mktemp -d)
trap 'rm -rf "${constraints_dir}"' EXIT
uv pip freeze "${uv_target[@]}" > "${constraints_dir}/installed.txt"
grep -E '^(torch|torchaudio|torchvision|triton|cuda-bindings|cuda-python|cuda-toolkit)==' \
    "${constraints_dir}/installed.txt" > "${constraints_dir}/constraints.txt"
uv pip install "${uv_target[@]}" -c "${constraints_dir}/constraints.txt" -r "${requirements_file}"

metadata=$(python3 - <<'PY'
import importlib.metadata as metadata

try:
    mooncake_version = metadata.version("mooncake-transfer-engine")
except metadata.PackageNotFoundError:
    mooncake_version = ""

print(metadata.version("nixl"), mooncake_version)
PY
)
read -r NIXL_VERSION MOONCAKE_VERSION <<<"${metadata}"
MOONCAKE_VERSION="${MOONCAKE_VERSION:-}"

# Both variants expose the same module; keep only the installed CUDA runtime's.
uv pip uninstall "${uv_target[@]}" nixl-cu12 nixl-cu13
uv pip install "${uv_target[@]}" --no-deps "nixl-cu${CUDA_MAJOR}==${NIXL_VERSION}"

# The default Mooncake wheel links libcudart.so.12. Both variants expose the
# same mooncake package, so remove it before installing the CUDA 13 build.
if [[ "${CUDA_MAJOR}" == "13" && -n "${MOONCAKE_VERSION}" ]]; then
    uv pip uninstall "${uv_target[@]}" mooncake-transfer-engine
    uv pip install "${uv_target[@]}" -c "${constraints_dir}/constraints.txt" \
        "mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}"
fi
