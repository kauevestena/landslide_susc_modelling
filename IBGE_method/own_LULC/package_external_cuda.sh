#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_DIR="${1:-${ROOT_DIR}/external_lulc_package}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  PACKAGE_DIR="${ROOT_DIR}/external_lulc_package"
  DRY_RUN=1
elif [[ "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

ORTHO_PATH="${LULC_INPUT_ORTHO:-/home/kaue/data/landslide/feb26/Ortho_4_GNSS-AAT_16cm.tif}"
POLYGONS_PATH="${LULC_INPUT_POLYGONS:-${ROOT_DIR}/IBGE_method/own_LULC/extra_data/Classes_Uso_Solo.gpkg}"
ARCHIVE_NAME="landslide_lulc_external_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "[lulc-package] root=${ROOT_DIR}"
echo "[lulc-package] package_dir=${PACKAGE_DIR}"
echo "[lulc-package] source_archive=${ARCHIVE_NAME}"
echo "[lulc-package] ortho=${ORTHO_PATH}"
echo "[lulc-package] polygons=${POLYGONS_PATH}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[lulc-package] dry run only; no files will be written"
  echo "[lulc-package] would create source archive excluding .venv, outputs, artifacts, caches, and generated products"
  echo "[lulc-package] would write external_data_manifest.json and external_env.sh.example"
  exit 0
fi

mkdir -p "${PACKAGE_DIR}"

PACKAGE_DIR="${PACKAGE_DIR}" ORTHO_PATH="${ORTHO_PATH}" POLYGONS_PATH="${POLYGONS_PATH}" \
  "${ROOT_DIR}/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


package_dir = Path(os.environ["PACKAGE_DIR"])
paths = {
    "orthophoto": Path(os.environ["ORTHO_PATH"]).expanduser(),
    "polygons": Path(os.environ["POLYGONS_PATH"]).expanduser(),
}
manifest = {"files": {}}
for key, path in paths.items():
    if not path.exists():
        raise SystemExit(f"missing {key}: {path}")
    manifest["files"][key] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
(package_dir / "external_data_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
)
(package_dir / "external_env.sh.example").write_text(
    "\n".join(
        [
            "# Source this file or pass paths to run_fullres_external.sh.",
            f"export LULC_INPUT_ORTHO='{paths['orthophoto'].resolve()}'",
            f"export LULC_INPUT_POLYGONS='{paths['polygons'].resolve()}'",
            "export LULC_OUTPUT_DIR='$PWD/IBGE_method/own_LULC/outputs_fullres'",
            "",
        ]
    ),
    encoding="utf-8",
)
PY

tar -czf "${PACKAGE_DIR}/${ARCHIVE_NAME}" \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='outputs' \
  --exclude='artifacts' \
  --exclude='DL_method/outputs' \
  --exclude='IBGE_method/outputs' \
  --exclude='SGB_method/outputs' \
  --exclude='IBGE_method/own_LULC/outputs' \
  --exclude='IBGE_method/own_LULC/outputs_fullres' \
  -C "${ROOT_DIR}" .

echo "[lulc-package] wrote ${PACKAGE_DIR}/${ARCHIVE_NAME}"
echo "[lulc-package] wrote ${PACKAGE_DIR}/external_data_manifest.json"
echo "[lulc-package] wrote ${PACKAGE_DIR}/external_env.sh.example"
