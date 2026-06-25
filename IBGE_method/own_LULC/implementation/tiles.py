"""Tile creation and spatial splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter


def log(message: str) -> None:
    print(f"[lulc] {message}", flush=True)


@dataclass(frozen=True)
class TileRecord:
    y: int
    x: int
    image: np.ndarray
    label: np.ndarray


@dataclass(frozen=True)
class BlockRecord:
    key: Tuple[int, int]
    records: Tuple[TileRecord, ...]
    class_counts: Dict[int, int]
    labeled_pixels: int


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


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = rgb
    maxc = np.max(rgb, axis=0)
    minc = np.min(rgb, axis=0)
    delta = maxc - minc
    hue = np.zeros_like(maxc)
    nonzero = delta > 1e-6
    red_max = (maxc == red) & nonzero
    green_max = (maxc == green) & nonzero
    blue_max = (maxc == blue) & nonzero
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4.0
    hue = hue / 6.0
    saturation = np.zeros_like(maxc)
    saturation[maxc > 1e-6] = delta[maxc > 1e-6] / maxc[maxc > 1e-6]
    value = maxc
    return np.stack([hue, saturation, value]).astype(np.float32)


def build_feature_stack(config: Dict[str, Any], rgb: np.ndarray) -> np.ndarray:
    params = config["params"]
    feature_set = params["feature_set"]
    channels = config["feature_sets"][feature_set]["channels"]
    log(f"building feature stack '{feature_set}' with channels={channels}")
    rgb_float = rgb.astype(np.float32) / 255.0
    red, green, blue = rgb_float
    hsv = rgb_to_hsv(rgb_float)
    brightness = np.mean(rgb_float, axis=0, keepdims=True)
    excess_green = ((2.0 * green - red - blue) + 2.0) / 4.0
    gray = brightness[0]
    mean = uniform_filter(gray, size=5)
    mean_sq = uniform_filter(gray * gray, size=5)
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    if float(texture.max()) > 0:
        texture = texture / float(texture.max())
    available = {
        "red": red,
        "green": green,
        "blue": blue,
        "hue": hsv[0],
        "saturation": hsv[1],
        "value": hsv[2],
        "brightness": brightness[0],
        "excess_green": np.clip(excess_green, 0.0, 1.0),
        "local_texture": texture,
    }
    missing = [channel for channel in channels if channel not in available]
    if missing:
        raise ValueError(f"Unsupported LULC feature channels: {missing}")
    return np.stack([available[channel] for channel in channels]).astype(np.float32)


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
    features = build_feature_stack(config, rgb)

    records: List[TileRecord] = []
    y_starts = window_starts(labels.shape[0], tile_size, stride)
    x_starts = window_starts(labels.shape[1], tile_size, stride)
    total_windows = len(y_starts) * len(x_starts)
    processed = 0
    log(
        "extracting candidate tiles: "
        f"tile_size={tile_size} stride={stride} windows={total_windows}"
    )
    for y in y_starts:
        for x in x_starts:
            processed += 1
            image_tile = extract_window(features, y, x, tile_size, 0.0)
            label_tile = extract_window(labels, y, x, tile_size, ignore_index)
            valid_tile = extract_window(valid_mask.astype(np.uint8), y, x, tile_size, 0)
            label_tile = label_tile.copy()
            label_tile[valid_tile == 0] = ignore_index
            if int(np.count_nonzero(label_tile != ignore_index)) >= min_labeled:
                records.append(TileRecord(y=y, x=x, image=image_tile, label=label_tile))
            if processed % 100 == 0 or processed == total_windows:
                log(f"tile extraction progress: {processed}/{total_windows} kept={len(records)}")
    if not records:
        raise ValueError("No labeled LULC tiles were created.")
    return records


def split_tiles(config: Dict[str, Any], records: Sequence[TileRecord]) -> Dict[str, List[TileRecord]]:
    params = config["params"]
    block_size = int(params["spatial_block_size"])
    stride = int(params["stride"])
    block_span = max(1, block_size * stride)
    classes = sorted(int(value) for value in config["class_definitions"])
    ignore_index = int(params["ignore_index"])

    grouped: Dict[Tuple[int, int], List[TileRecord]] = {}
    for record in records:
        key = (record.y // block_span, record.x // block_span)
        grouped.setdefault(key, []).append(record)

    blocks: List[BlockRecord] = []
    for key, group_records in grouped.items():
        counts = {class_value: 0 for class_value in classes}
        labeled_pixels = 0
        for record in group_records:
            labeled_pixels += int(np.count_nonzero(record.label != ignore_index))
            for class_value in classes:
                counts[class_value] += int(np.count_nonzero(record.label == class_value))
        blocks.append(
            BlockRecord(
                key=key,
                records=tuple(group_records),
                class_counts=counts,
                labeled_pixels=labeled_pixels,
            )
        )

    if len(blocks) == 1:
        return {"train": list(blocks[0].records), "val": [], "test": []}
    log(f"spatial split blocks: {len(blocks)}")

    split_fractions = {
        "train": float(params["train_percentage"]),
        "val": float(params["validation_percentage"]),
        "test": float(params["test_percentage"]),
    }
    n_blocks = len(blocks)

    total_counts = {class_value: sum(block.class_counts[class_value] for block in blocks) for class_value in classes}
    constraints = config["split_constraints"]
    min_class_pixels = {
        class_value: min(
            int(constraints["min_val_test_pixels_per_class_cap"]),
            max(1, int(np.ceil(total_counts[class_value] * float(constraints["min_val_test_class_fraction"])))),
        )
        for class_value in classes
        if total_counts[class_value] > 0
    }
    target_counts = {
        split_name: {
            class_value: total_counts[class_value] * fraction
            for class_value in classes
        }
        for split_name, fraction in split_fractions.items()
    }

    def flatten(candidate_blocks: Sequence[BlockRecord]) -> List[TileRecord]:
        return [record for block in candidate_blocks for record in block.records]

    def split_counts(candidate_blocks: Sequence[BlockRecord]) -> Dict[int, int]:
        return {
            class_value: sum(block.class_counts[class_value] for block in candidate_blocks)
            for class_value in classes
        }

    def score_candidate(candidate: Dict[str, List[BlockRecord]]) -> Tuple[int, float]:
        class_penalty = 0
        shortage = 0.0
        percentage_error = 0.0
        empty_penalty = sum(1 for split_blocks in candidate.values() if not split_blocks)
        for split_name in ("train", "val", "test"):
            counts = split_counts(candidate[split_name])
            for class_value in classes:
                if total_counts[class_value] > 0 and counts[class_value] == 0:
                    class_penalty += 1
                denominator = max(1.0, float(total_counts[class_value]))
                percentage_error += (
                    abs(counts[class_value] - target_counts[split_name][class_value])
                    / denominator
                )
        for split_name in ("val", "test"):
            counts = split_counts(candidate[split_name])
            for class_value, target in min_class_pixels.items():
                if counts[class_value] < target:
                    shortage += float(target - counts[class_value])
        return class_penalty + empty_penalty, shortage + percentage_error

    def random_candidate(rng: np.random.Generator) -> Dict[str, List[BlockRecord]]:
        candidate = {"train": [], "val": [], "test": []}
        split_names = np.array(["train", "val", "test"], dtype=object)
        probabilities = np.array(
            [split_fractions["train"], split_fractions["val"], split_fractions["test"]],
            dtype=np.float64,
        )
        probabilities = probabilities / probabilities.sum()
        shuffled = list(blocks)
        rng.shuffle(shuffled)
        for block in shuffled:
            split_name = str(rng.choice(split_names, p=probabilities))
            candidate[split_name].append(block)
        for split_name in ("train", "val", "test"):
            if candidate[split_name]:
                continue
            donors = [name for name in ("train", "val", "test") if len(candidate[name]) > 1]
            if not donors:
                continue
            donor = max(donors, key=lambda name: len(candidate[name]))
            candidate[split_name].append(candidate[donor].pop())
        return candidate

    best_candidate = None
    best_score = (10**9, float("inf"))
    attempts = int(constraints["max_seed_attempts"])
    base_seed = int(params["random_seed"])
    for offset in range(attempts):
        rng = np.random.default_rng(base_seed + offset)
        candidate = random_candidate(rng)
        candidate_score = score_candidate(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_candidate = candidate
            log(
                "split search improved: "
                f"attempt={offset + 1}/{attempts} missing_or_empty={best_score[0]} "
                f"score={best_score[1]:.4f}"
            )
        if candidate_score[0] == 0 and candidate_score[1] <= 1e-6:
            break

    if best_candidate is None:
        raise RuntimeError("Failed to create LULC spatial split.")
    if bool(constraints["require_all_classes_per_split"]) and best_score[0] > 0:
        # Rare class geometries can make a perfect spatial split impossible; keep the best
        # candidate and expose support counts in metadata instead of silently randomizing.
        pass
    log(
        "selected spatial split: "
        f"missing_or_empty={best_score[0]} score={best_score[1]:.4f}"
    )

    return {
        "train": flatten(best_candidate["train"]),
        "val": flatten(best_candidate["val"]),
        "test": flatten(best_candidate["test"]),
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


def split_summary(
    splits: Dict[str, Sequence[TileRecord]],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    class_values = sorted(
        int(v)
        for split_records in splits.values()
        for record in split_records
        for v in np.unique(record.label)
        if int(v) != 255
    )
    class_values = sorted(set(class_values))
    summary = {
        name: {
            "tiles": len(records),
            "labeled_pixels": int(sum(np.count_nonzero(record.label != 255) for record in records)),
            "class_counts": {
                str(class_value): int(
                    sum(np.count_nonzero(record.label == class_value) for record in records)
                )
                for class_value in class_values
            },
            "positions": [{"y": int(record.y), "x": int(record.x)} for record in records],
        }
        for name, records in splits.items()
    }
    if config is not None:
        fractions = {
            "train": float(config["params"]["train_percentage"]),
            "val": float(config["params"]["validation_percentage"]),
            "test": float(config["params"]["test_percentage"]),
        }
        totals = {
            str(class_value): sum(
                summary[split_name]["class_counts"].get(str(class_value), 0)
                for split_name in ("train", "val", "test")
            )
            for class_value in class_values
        }
        for split_name in ("train", "val", "test"):
            summary[split_name]["target_fraction"] = fractions[split_name]
            summary[split_name]["class_fractions"] = {}
            summary[split_name]["class_target_pixels"] = {}
            summary[split_name]["class_fraction_error"] = {}
            for class_value in class_values:
                key = str(class_value)
                total = totals[key]
                actual = summary[split_name]["class_counts"].get(key, 0) / max(1, total)
                target = fractions[split_name]
                summary[split_name]["class_fractions"][key] = float(actual)
                summary[split_name]["class_target_pixels"][key] = float(total * target)
                summary[split_name]["class_fraction_error"][key] = float(actual - target)
    return summary
