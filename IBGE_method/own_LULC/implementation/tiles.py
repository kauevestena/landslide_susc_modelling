"""Tile creation and spatial splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TileRecord:
    y: int
    x: int
    image: np.ndarray
    label: np.ndarray


def window_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(length - tile_size + 1, 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_window(
    array: np.ndarray,
    y: int,
    x: int,
    tile_size: int,
    fill_value: float,
) -> np.ndarray:
    if array.ndim == 3:
        channels, height, width = array.shape
        out = np.full((channels, tile_size, tile_size), fill_value, dtype=array.dtype)
        src = array[:, y : min(y + tile_size, height), x : min(x + tile_size, width)]
        out[:, : src.shape[1], : src.shape[2]] = src
        return out
    height, width = array.shape
    out = np.full((tile_size, tile_size), fill_value, dtype=array.dtype)
    src = array[y : min(y + tile_size, height), x : min(x + tile_size, width)]
    out[: src.shape[0], : src.shape[1]] = src
    return out


def build_tiles(
    config: Dict[str, Any],
    rgb: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
) -> List[TileRecord]:
    params = config["params"]
    tile_size = int(params["tile_size"])
    stride = int(params["stride"])
    ignore_index = int(params["ignore_index"])
    min_labeled = int(params["min_labeled_pixels_per_tile"])
    features = rgb.astype(np.float32) / 255.0

    records: List[TileRecord] = []
    for y in window_starts(labels.shape[0], tile_size, stride):
        for x in window_starts(labels.shape[1], tile_size, stride):
            image_tile = extract_window(features, y, x, tile_size, 0.0)
            label_tile = extract_window(labels, y, x, tile_size, ignore_index)
            valid_tile = extract_window(valid_mask.astype(np.uint8), y, x, tile_size, 0)
            label_tile = label_tile.copy()
            label_tile[valid_tile == 0] = ignore_index
            if int(np.count_nonzero(label_tile != ignore_index)) >= min_labeled:
                records.append(TileRecord(y=y, x=x, image=image_tile, label=label_tile))
    if not records:
        raise ValueError("No labeled LULC tiles were created.")
    return records


def split_tiles(config: Dict[str, Any], records: Sequence[TileRecord]) -> Dict[str, List[TileRecord]]:
    params = config["params"]
    rng = np.random.default_rng(int(params["random_seed"]))
    block_size = int(params["spatial_block_size"])
    stride = int(params["stride"])
    block_span = max(1, block_size * stride)

    grouped: Dict[Tuple[int, int], List[TileRecord]] = {}
    for record in records:
        key = (record.y // block_span, record.x // block_span)
        grouped.setdefault(key, []).append(record)

    groups = list(grouped.values())
    rng.shuffle(groups)
    if len(groups) == 1:
        return {"train": list(groups[0]), "val": [], "test": []}

    train_fraction = float(params["train_percentage"])
    val_fraction = float(params["validation_percentage"])
    n_groups = len(groups)
    n_train = max(1, int(round(n_groups * train_fraction)))
    n_val = int(round(n_groups * val_fraction))
    if n_train + n_val >= n_groups:
        n_val = max(0, n_groups - n_train - 1)
    n_test = n_groups - n_train - n_val
    if n_test < 0:
        n_test = 0

    train_groups = groups[:n_train]
    val_groups = groups[n_train : n_train + n_val]
    test_groups = groups[n_train + n_val :]
    return {
        "train": [record for group in train_groups for record in group],
        "val": [record for group in val_groups for record in group],
        "test": [record for group in test_groups for record in group],
    }


def limit_split_for_smoke(
    config: Dict[str, Any], splits: Dict[str, List[TileRecord]]
) -> Dict[str, List[TileRecord]]:
    smoke = config["params"]["smoke_test"]
    return {
        "train": splits["train"][: int(smoke["max_train_tiles"])],
        "val": splits["val"][: int(smoke["max_eval_tiles"])],
        "test": splits["test"][: int(smoke["max_eval_tiles"])],
    }


def split_summary(splits: Dict[str, Sequence[TileRecord]]) -> Dict[str, Any]:
    return {
        name: {
            "tiles": len(records),
            "positions": [{"y": int(record.y), "x": int(record.x)} for record in records],
        }
        for name, records in splits.items()
    }
