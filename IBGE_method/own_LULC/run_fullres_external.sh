#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORTHO_PATH="${1:-}"
OUTPUT_DIR="${2:-${ROOT_DIR}/IBGE_method/own_LULC/outputs_fullres}"
POLYGONS_PATH="${3:-${ROOT_DIR}/IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg}"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

usage() {
  cat <<'USAGE'
Usage:
  bash IBGE_method/own_LULC/run_fullres_external.sh /path/to/orthophoto.tif
  bash IBGE_method/own_LULC/run_fullres_external.sh /path/to/orthophoto.tif /path/to/output_dir /path/to/Classes_Uso_Solo.gpkg

Environment toggles:
  LULC_FORCE=1          rerun completed experiments
  LULC_SMOKE=1          run configured smoke mode
  LULC_NO_PROMOTE=1     do not promote the final ensemble
  LULC_SKIP_PREFLIGHT=1 skip CUDA/data preflight
USAGE
}

if [[ -z "${ORTHO_PATH}" || "${ORTHO_PATH}" == "-h" || "${ORTHO_PATH}" == "--help" ]]; then
  usage
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[lulc-external] ERROR: missing ${PYTHON_BIN}; run setup_external_cuda_venv.sh first" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${OUTPUT_DIR}/logs/fullres_lulc_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

export LULC_INPUT_ORTHO="$(realpath "${ORTHO_PATH}")"
export LULC_INPUT_POLYGONS="$(realpath "${POLYGONS_PATH}")"
export LULC_OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"

echo "[lulc-external] root=${ROOT_DIR}"
echo "[lulc-external] ortho=${LULC_INPUT_ORTHO}"
echo "[lulc-external] polygons=${LULC_INPUT_POLYGONS}"
echo "[lulc-external] output=${LULC_OUTPUT_DIR}"
echo "[lulc-external] log=${LOG_FILE}"

if [[ "${LULC_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/IBGE_method/own_LULC/preflight_external_cuda.py" \
    --ortho "${LULC_INPUT_ORTHO}" \
    --polygons "${LULC_INPUT_POLYGONS}" \
    --output-dir "${LULC_OUTPUT_DIR}" \
    --target-resolution 0.16
fi

PIPELINE_ARGS=(--fullres-sweep)
if [[ "${LULC_FORCE:-0}" == "1" ]]; then
  PIPELINE_ARGS+=(--force)
fi
if [[ "${LULC_SMOKE:-0}" == "1" ]]; then
  PIPELINE_ARGS+=(--smoke)
  PIPELINE_ARGS+=(--no-promote)
fi
if [[ "${LULC_NO_PROMOTE:-0}" == "1" ]]; then
  PIPELINE_ARGS+=(--no-promote)
fi

echo "[lulc-external] running pipeline: ${PIPELINE_ARGS[*]}"
"${PYTHON_BIN}" -m IBGE_method.own_LULC.implementation.pipeline "${PIPELINE_ARGS[@]}"

if [[ "${LULC_SMOKE:-0}" == "1" ]]; then
  echo "[lulc-external] smoke run complete; skipping ensemble promotion"
  exit 0
fi

echo "[lulc-external] rebuilding/promoting full-resolution ensemble"
ENSEMBLE_ARGS=(--ensemble-only --target-resolution 0.16)
if [[ "${LULC_NO_PROMOTE:-0}" == "1" ]]; then
  ENSEMBLE_ARGS+=(--no-promote)
fi
"${PYTHON_BIN}" -m IBGE_method.own_LULC.implementation.pipeline "${ENSEMBLE_ARGS[@]}"

echo "[lulc-external] complete"
