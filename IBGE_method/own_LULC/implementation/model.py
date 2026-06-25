"""Model training and inference for custom LULC segmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import output_path
from .metrics import confusion_metrics
from .tiles import TileRecord, build_feature_stack, extract_window, window_starts


def log(message: str) -> None:
    print(f"[lulc] {message}", flush=True)


class LULCTileDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TileRecord],
        class_values: Sequence[int],
        ignore_index: int,
        augment_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.records = list(records)
        self.class_values = list(class_values)
        self.ignore_index = int(ignore_index)
        self.augment_config = augment_config or {}
        self.value_to_index = {value: idx for idx, value in enumerate(self.class_values)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        image = record.image.copy()
        label = record.label.copy()
        image, label = self._augment(image, label)
        encoded = np.full(label.shape, self.ignore_index, dtype=np.int64)
        for class_value, class_idx in self.value_to_index.items():
            encoded[label == class_value] = class_idx
        return (
            torch.from_numpy(np.ascontiguousarray(image.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(encoded)),
        )

    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.augment_config
        if not cfg:
            return image, label
        if np.random.rand() < float(cfg["flip_probability"]):
            image = image[:, :, ::-1]
            label = label[:, ::-1]
        if np.random.rand() < float(cfg["flip_probability"]):
            image = image[:, ::-1, :]
            label = label[::-1, :]
        if bool(cfg["rotate90"]):
            k = int(np.random.randint(0, 4))
            image = np.rot90(image, k=k, axes=(1, 2))
            label = np.rot90(label, k=k)
        brightness = float(cfg["brightness"])
        if brightness > 0:
            image = image + np.random.uniform(-brightness, brightness)
        contrast = float(cfg["contrast"])
        if contrast > 0:
            image = image * np.random.uniform(1.0 - contrast, 1.0 + contrast)
        if bool(cfg["blur_enabled"]):
            sigma = max(0.1, float(cfg["blur_kernel_size"]) / 6.0)
            image = gaussian_filter(image, sigma=(0.0, sigma, sigma))
        if bool(cfg["noise_enabled"]):
            image = image + np.random.normal(0.0, float(cfg["noise_std"]), size=image.shape)
        return np.clip(image, 0.0, 1.0), label.copy()


class ConfiguredSegmentationLoss(nn.Module):
    def __init__(
        self,
        name: str,
        num_classes: int,
        ignore_index: int,
        class_weights: Optional[torch.Tensor],
        loss_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.name = name
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.loss_cfg = loss_cfg
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, ignore_index=ignore_index
        )
        self.focal = smp.losses.FocalLoss(
            mode="multiclass",
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            ignore_index=ignore_index,
        )
        self.lovasz = smp.losses.LovaszLoss(
            mode="multiclass", ignore_index=ignore_index, from_logits=True
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.name == "weighted_ce_dice":
            return (
                float(self.loss_cfg["cross_entropy_weight"]) * self.ce(logits, labels)
                + float(self.loss_cfg["dice_weight"]) * self.dice(logits, labels)
            )
        if self.name == "focal_dice":
            return (
                float(self.loss_cfg["focal_weight"]) * self.focal(logits, labels)
                + float(self.loss_cfg["dice_weight"]) * self.dice(logits, labels)
            )
        if self.name == "weighted_ce_lovasz":
            return (
                float(self.loss_cfg["cross_entropy_weight"]) * self.ce(logits, labels)
                + float(self.loss_cfg["lovasz_weight"]) * self.lovasz(logits, labels)
            )
        if self.name == "focal_lovasz":
            return (
                float(self.loss_cfg["focal_weight"]) * self.focal(logits, labels)
                + float(self.loss_cfg["lovasz_weight"]) * self.lovasz(logits, labels)
            )
        raise ValueError(f"Unsupported LULC loss: {self.name}")


def select_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        try:
            test_input = torch.randn(1, 3, 16, 16).cuda()
            test_conv = torch.nn.Conv2d(3, 3, 3).cuda()
            _ = test_conv(test_input)
            log("using CUDA device for LULC model")
            return torch.device("cuda")
        except (RuntimeError, AssertionError):
            log("CUDA requested but unavailable/incompatible; using CPU")
            return torch.device("cpu")
    if use_cuda:
        log("CUDA requested but torch.cuda is unavailable; using CPU")
    else:
        log("CUDA disabled by config; using CPU")
    return torch.device("cpu")


def create_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["params"]["model"]
    constructors = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
    }
    architecture = model_cfg["architecture"].lower()
    if architecture not in constructors:
        raise ValueError(f"Unsupported LULC architecture: {model_cfg['architecture']}")
    return constructors[architecture](
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=int(model_cfg["input_channels"]),
        classes=int(model_cfg["output_classes"]),
    )


def compute_class_weights(
    records: Sequence[TileRecord],
    class_values: Sequence[int],
    ignore_index: int,
    strategy: str,
) -> Optional[torch.Tensor]:
    if strategy == "none":
        return None
    if strategy != "inverse_frequency":
        raise ValueError(f"Unsupported class_weight_strategy: {strategy}")
    counts = np.zeros(len(class_values), dtype=np.float64)
    value_to_index = {value: idx for idx, value in enumerate(class_values)}
    for record in records:
        for class_value, class_idx in value_to_index.items():
            counts[class_idx] += np.count_nonzero(record.label == class_value)
    if not bool(np.any(counts)):
        return None
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(class_values) * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(
    records: Sequence[TileRecord],
    config: Dict[str, Any],
    class_values: Sequence[int],
    training: bool,
) -> Optional[DataLoader]:
    if not records:
        return None
    params = config["params"]
    dataset = LULCTileDataset(
        records,
        class_values,
        int(params["ignore_index"]),
        params["augmentation"] if training else None,
    )
    sampler = None
    shuffle = training
    if training and config["sampler"]["strategy"] == "class_balanced_weighted":
        weights = compute_tile_sampling_weights(
            records,
            class_values,
            int(params["ignore_index"]),
            float(config["sampler"]["rare_class_power"]),
            float(config["sampler"]["max_weight"]),
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(params["training"]["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(params["training"]["num_workers"]),
        pin_memory=False,
    )


def compute_tile_sampling_weights(
    records: Sequence[TileRecord],
    class_values: Sequence[int],
    ignore_index: int,
    rare_class_power: float,
    max_weight: float,
) -> List[float]:
    counts = {class_value: 0.0 for class_value in class_values}
    for record in records:
        for class_value in class_values:
            counts[class_value] += float(np.count_nonzero(record.label == class_value))
    total = sum(counts.values())
    if total <= 0:
        return [1.0 for _ in records]
    class_weights = {
        class_value: (total / max(1.0, counts[class_value])) ** rare_class_power
        for class_value in class_values
    }
    weights = []
    for record in records:
        labeled = float(np.count_nonzero(record.label != ignore_index))
        if labeled <= 0:
            weights.append(1.0)
            continue
        score = 0.0
        for class_value in class_values:
            score += class_weights[class_value] * float(np.count_nonzero(record.label == class_value))
        weights.append(float(min(max_weight, max(1.0, score / labeled))))
    return weights


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    if loader is None:
        return {
            "loss": 0.0,
            "metrics": confusion_metrics(np.zeros((num_classes, num_classes), dtype=np.int64)),
            "batches": 0,
        }
    model.eval()
    total_loss = 0.0
    batches = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item())
            batches += 1
            preds = torch.argmax(logits, dim=1)
            valid = labels != loss_fn.ignore_index
            labels_np = labels[valid].detach().cpu().numpy()
            preds_np = preds[valid].detach().cpu().numpy()
            for true_idx in range(num_classes):
                for pred_idx in range(num_classes):
                    confusion[true_idx, pred_idx] += int(
                        np.count_nonzero((labels_np == true_idx) & (preds_np == pred_idx))
                    )
    return {
        "loss": total_loss / batches if batches else 0.0,
        "metrics": confusion_metrics(confusion),
        "batches": batches,
    }


def train_model(
    config: Dict[str, Any],
    splits: Dict[str, Sequence[TileRecord]],
) -> Tuple[nn.Module, Dict[str, Any]]:
    params = config["params"]
    training_cfg = params["training"]
    class_values = sorted(int(v) for v in config["class_definitions"].keys())
    torch.manual_seed(int(params["random_seed"]))
    np.random.seed(int(params["random_seed"]))
    device = select_device(bool(training_cfg["use_cuda"]))

    log(
        "creating model: "
        f"{params['model']['architecture']}/{params['model']['encoder']} "
        f"in_channels={params['model']['input_channels']} "
        f"classes={params['model']['output_classes']}"
    )
    model = create_model(config).to(device)
    weights = compute_class_weights(
        splits["train"],
        class_values,
        int(params["ignore_index"]),
        training_cfg["class_weight_strategy"],
    )
    if weights is not None:
        weights = weights.to(device)
    log(f"class weights: {weights.detach().cpu().tolist() if weights is not None else []}")
    loss_fn = ConfiguredSegmentationLoss(
        name=params["loss"]["name"],
        num_classes=len(class_values),
        ignore_index=int(params["ignore_index"]),
        class_weights=weights,
        loss_cfg=params["loss"],
    )
    if training_cfg["optimizer"].lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {training_cfg['optimizer']}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = None
    if training_cfg["scheduler"].lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(training_cfg["epochs"]))
        )
    elif training_cfg["scheduler"].lower() != "none":
        raise ValueError(f"Unsupported scheduler: {training_cfg['scheduler']}")

    train_loader = make_loader(splits["train"], config, class_values, training=True)
    val_loader = make_loader(splits["val"], config, class_values, training=False)
    test_loader = make_loader(splits["test"], config, class_values, training=False)
    if train_loader is None:
        raise ValueError("No training tiles available.")
    log(
        "training loaders: "
        f"train_tiles={len(splits['train'])} val_tiles={len(splits['val'])} "
        f"test_tiles={len(splits['test'])} batch_size={training_cfg['batch_size']}"
    )

    best_score = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    best_metric_name = training_cfg["save_best_metric"]

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        log(f"epoch {epoch}/{training_cfg['epochs']} started")
        model.train()
        train_loss = 0.0
        train_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1
        if scheduler is not None:
            scheduler.step()

        val_result = evaluate(model, val_loader, loss_fn, device, len(class_values))
        val_metrics = val_result["metrics"] or {}
        if val_metrics:
            score = float(val_metrics.get(best_metric_name, float("nan")))
            if not np.isfinite(score):
                score = -float(val_result["loss"])
        else:
            score = -(train_loss / max(1, train_batches))
        if not np.isfinite(score):
            score = 0.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(1, train_batches),
                "val_loss": val_result["loss"],
                "val_metrics": val_result["metrics"],
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            improved = True
        else:
            epochs_without_improvement += 1
            improved = False
        log(
            f"epoch {epoch}/{training_cfg['epochs']} done: "
            f"train_loss={train_loss / max(1, train_batches):.6f} "
            f"val_loss={val_result['loss']:.6f} "
            f"{best_metric_name}={score:.6f} "
            f"best={best_score:.6f} "
            f"improved={improved} "
            f"patience={epochs_without_improvement}/{training_cfg['early_stopping_patience']}"
        )
        if epochs_without_improvement >= int(training_cfg["early_stopping_patience"]):
            log(f"early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = output_path(config, "model_filename")
    log(f"saving best model: {model_path}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_values": class_values,
            "params": params,
        },
        model_path,
    )
    test_result = evaluate(model, test_loader, loss_fn, device, len(class_values))
    log(
        "test metrics: "
        f"loss={test_result['loss']:.6f} "
        f"macro_iou={test_result['metrics']['macro_iou']:.6f} "
        f"overall_accuracy={test_result['metrics']['overall_accuracy']:.6f}"
    )
    return model, {
        "device": str(device),
        "class_values": class_values,
        "class_weights": weights.detach().cpu().tolist() if weights is not None else [],
        "best_score": best_score,
        "best_epoch": int(max(history, key=lambda item: item["score"])["epoch"]) if history else 0,
        "loss": params["loss"],
        "model": params["model"],
        "feature_set": params["feature_set"],
        "history": history,
        "test": test_result,
        "model_path": str(model_path),
    }


def infer_full_raster(
    config: Dict[str, Any],
    model: nn.Module,
    rgb: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    params = config["params"]
    inference_cfg = params["inference"]
    post_cfg = params["postprocessing"]
    class_values = sorted(int(v) for v in config["class_definitions"].keys())
    device = select_device(bool(inference_cfg["use_cuda"]))
    model = model.to(device)
    model.eval()

    features = build_feature_stack(config, rgb)
    window_size = int(inference_cfg["window_size"])
    overlap = int(inference_cfg["overlap"])
    stride = max(1, window_size - overlap)
    batch_size = int(inference_cfg["batch_size"])
    height, width = features.shape[1:]
    prob_sum = np.zeros((len(class_values), height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    pending: List[Tuple[int, int, np.ndarray]] = []
    y_starts = window_starts(height, window_size, stride)
    x_starts = window_starts(width, window_size, stride)
    total_windows = len(y_starts) * len(x_starts)
    processed_windows = 0
    log(
        "inference windows: "
        f"grid={width}x{height} window={window_size} overlap={overlap} "
        f"stride={stride} windows={total_windows} batch_size={batch_size}"
    )

    def flush() -> None:
        if not pending:
            return
        batch = torch.from_numpy(np.stack([item[2] for item in pending])).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).detach().cpu().numpy()
        nonlocal processed_windows
        for (y, x, _tile), prob in zip(pending, probs):
            y1 = min(y + window_size, height)
            x1 = min(x + window_size, width)
            prob_sum[:, y:y1, x:x1] += prob[:, : y1 - y, : x1 - x]
            weight_sum[y:y1, x:x1] += 1.0
            processed_windows += 1
        if processed_windows % 25 == 0 or processed_windows == total_windows:
            log(f"inference progress: {processed_windows}/{total_windows} windows")
        pending.clear()

    for y in y_starts:
        for x in x_starts:
            pending.append((y, x, extract_window(features, y, x, window_size, 0.0)))
            if len(pending) >= batch_size:
                flush()
    flush()

    probabilities = prob_sum / np.maximum(weight_sum[None, :, :], 1.0)
    if bool(post_cfg["smoothing_enabled"]):
        sigma = float(post_cfg["smoothing_sigma"])
        probabilities = np.stack(
            [gaussian_filter(probabilities[idx], sigma=sigma) for idx in range(probabilities.shape[0])]
        )
        probabilities = probabilities / np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-6)

    pred_idx = np.argmax(probabilities, axis=0)
    confidence = np.max(probabilities, axis=0)
    lulc = np.take(np.array(class_values, dtype=np.uint8), pred_idx).astype(np.uint8)
    lulc[~valid_mask] = int(params["output_nodata"])
    min_conf = float(post_cfg["minimum_mapping_confidence"])
    if min_conf > 0:
        lulc[confidence < min_conf] = int(params["output_nodata"])
    return lulc, probabilities.astype(np.float32)
