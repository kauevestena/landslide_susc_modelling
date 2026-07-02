"""Generate the IBGE lite product using the best single full-res LULC model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import rasterio

from src.three_method_comparison import MethodDirs, ROOT, generate_ibge_method


FULLRES_OUTPUTS = ROOT / "IBGE_method" / "own_LULC" / "outputs_fullres"
DEFAULT_DTM = Path("/home/kaue/data/landslide/dtm_final.tif")
LITE_ROOT = ROOT / "IBGE_method"
LITE_OUTPUTS = LITE_ROOT / "outputs_lite"
LITE_REPORTS = LITE_ROOT / "reports_lite"
LITE_CONFIGS = LITE_ROOT / "configs_lite"
LITE_WEBSITE = LITE_ROOT / "website_lite"
SELECTION_FILENAME = "lulc_single_model_selection.json"


def log(message: str) -> None:
    print(f"[ibge-lite] {message}", flush=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    log(f"wrote JSON: {path}")


@contextmanager
def temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    previous: Dict[str, Optional[str]] = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def run_sort_value(row: Mapping[str, Any]) -> Tuple[float, float, str]:
    return (
        float(row.get("val_macro_iou", -1.0) or -1.0),
        float(row.get("test_macro_iou", -1.0) or -1.0),
        str(row.get("run_id", "")),
    )


def eligible_fullres_runs(runs: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    eligible: list[Dict[str, Any]] = []
    for row in runs:
        run_id = str(row.get("run_id", ""))
        if run_id.startswith("smoke_"):
            continue
        try:
            resolution = float(row.get("resolution", 0.0))
        except (TypeError, ValueError):
            continue
        if abs(resolution - 0.16) > 1e-9:
            continue
        if "ensemble" in run_id.lower():
            continue
        eligible.append(dict(row))
    eligible.sort(key=run_sort_value, reverse=True)
    return eligible


def select_best_single_run(sweep_path: Path) -> Dict[str, Any]:
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing LULC sweep results: {sweep_path}")
    payload = read_json(sweep_path)
    runs = list(payload.get("runs", []))
    eligible = eligible_fullres_runs(runs)
    if not eligible:
        raise RuntimeError(f"No eligible full-resolution single-model LULC runs found in {sweep_path}")
    selected = eligible[0]
    log(
        "selected single LULC run: "
        f"{selected['run_id']} val_macro_iou={selected.get('val_macro_iou')} "
        f"test_macro_iou={selected.get('test_macro_iou')}"
    )
    return selected


def expected_lulc_path(run: Mapping[str, Any]) -> Path:
    output_dir = Path(str(run["output_dir"])).expanduser().resolve()
    return output_dir / "lulc_custom_16cm.tif"


def require_selected_lulc(run: Mapping[str, Any]) -> Path:
    path = expected_lulc_path(run)
    if path.exists():
        return path
    raise FileNotFoundError(
        "The best single-model LULC raster is not available locally.\n"
        f"Selected run: {run.get('run_id')}\n"
        f"Expected raster: {path}\n"
        "Copy the full-resolution individual run artifacts from the external CUDA machine, "
        "then rerun `.venv/bin/python -m src.ibge_lite --force`. The lite workflow will not "
        "fall back to the ensemble raster."
    )


def lite_dirs() -> MethodDirs:
    for path in (LITE_OUTPUTS, LITE_REPORTS, LITE_CONFIGS, LITE_WEBSITE):
        path.mkdir(parents=True, exist_ok=True)
    return MethodDirs(root=LITE_ROOT, outputs=LITE_OUTPUTS, configs=LITE_CONFIGS, reports=LITE_REPORTS)


def write_selection(run: Mapping[str, Any], lulc_path: Path, dtm_path: Path) -> Path:
    payload = {
        "selection_type": "best_single_model",
        "selection_metric": "val_macro_iou",
        "tie_breakers": ["test_macro_iou", "run_id"],
        "selected_run_id": run.get("run_id"),
        "selected_run": dict(run),
        "selected_lulc_path": str(lulc_path),
        "dtm_path": str(dtm_path),
        "ensemble_used": False,
        "note": "IBGE lite uses the best individual full-resolution LULC model, not the ensemble.",
    }
    report_path = LITE_REPORTS / SELECTION_FILENAME
    config_path = LITE_CONFIGS / SELECTION_FILENAME
    write_json(report_path, payload)
    write_json(config_path, payload)
    return report_path


def generate_website(selection_path: Path, lulc_path: Path, dtm_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "IBGE_WEBSITE_VARIANT": "lite",
            "LULC_REPORT_MODE": "best_single",
            "IBGE_WEBSITE_OUTPUTS_DIR": str(LITE_OUTPUTS),
            "IBGE_WEBSITE_REPORTS_DIR": str(LITE_REPORTS),
            "IBGE_WEBSITE_CONFIGS_DIR": str(LITE_CONFIGS),
            "IBGE_WEBSITE_DIR": str(LITE_WEBSITE),
            "IBGE_LULC_SINGLE_SELECTION_PATH": str(selection_path),
            "IBGE_CUSTOM_LULC_PATH": str(lulc_path),
            "IBGE_DTM_PATH": str(dtm_path),
        }
    )
    log(f"generating lite website: {LITE_WEBSITE}")
    subprocess.run([sys.executable, "-m", "src.ibge_website"], cwd=ROOT, env=env, check=True)


def generate_lite(force: bool = False) -> Dict[str, Any]:
    sweep_path = FULLRES_OUTPUTS / "sweep_results.json"
    selected = select_best_single_run(sweep_path)
    lulc_path = require_selected_lulc(selected)
    dtm_path = Path(os.environ.get("IBGE_DTM_PATH", DEFAULT_DTM)).expanduser().resolve()
    if not dtm_path.exists():
        raise FileNotFoundError(f"Lite DTM not found: {dtm_path}")

    if not force and (LITE_REPORTS / "summary.json").exists() and (LITE_WEBSITE / "index.html").exists():
        log("lite outputs already exist; use --force to regenerate")
        return {
            "selected_run": selected,
            "selected_lulc_path": str(lulc_path),
            "website": str(LITE_WEBSITE / "index.html"),
            "regenerated": False,
        }

    dirs = lite_dirs()
    env = {
        "IBGE_CUSTOM_LULC_PATH": str(lulc_path),
        "IBGE_DTM_PATH": str(dtm_path),
    }
    log(f"generating IBGE lite outputs with LULC: {lulc_path}")
    with temporary_env(env):
        with rasterio.open(dtm_path) as reference:
            summary = generate_ibge_method(reference, dirs)

    selection_path = write_selection(selected, lulc_path, dtm_path)
    generate_website(selection_path, lulc_path, dtm_path)
    return {
        "selected_run": selected,
        "selected_lulc_path": str(lulc_path),
        "summary": summary,
        "website": str(LITE_WEBSITE / "index.html"),
        "regenerated": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate lite IBGE outputs and website")
    args = parser.parse_args(argv)
    try:
        result = generate_lite(force=args.force)
    except (FileNotFoundError, RuntimeError) as exc:
        log(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
