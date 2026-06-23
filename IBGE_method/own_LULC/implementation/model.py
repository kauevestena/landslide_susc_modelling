"""Model training and inference for custom LULC segmentation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset

from .config import output_path
from .metrics import confusion_metrics
from .tiles import TileRecord, extract_window, window_starts


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


class CombinedCrossEntropyDice(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        class_weights: Optional[torch.Tensor],
        cross_entropy_weight: float,
        dice_weight: float,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, labels)
        valid = labels != self.ignore_index
        if not bool(valid.any()):
            return ce
        probs = torch.softmax(logits, dim=1)
        clamped = labels.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)
        dice = 1.0 - torch.mean((2.0 * intersection + 1e-6) / (denominator + 1e-6))
        return self.cross_entropy_weight * ce + self.dice_weight * dice


def select_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        try:
            test_input = torch.randn(1, 3, 16, 16).cuda()
            test_conv = torch.nn.Conv2d(3, 3, 3).cuda()
            _ = test_conv(test_input)
            return torch.device("cuda")
        except (RuntimeError, AssertionError):
            return torch.device("cpu")
    return torch.device("cpu")


def create_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config["params"]["model"]
    if model_cfg["architecture"].lower() != "unet":
        raise ValueError(f"Unsupported LULC architecture: {model_cfg['architecture']}")
    return smp.Unet(
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
    return DataLoader(
        dataset,
        batch_size=int(params["training"]["batch_size"]),
        shuffle=training,
        num_workers=int(params["training"]["num_workers"]),
        pin_memory=False,
    )


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

    model = create_model(config).to(device)
    weights = compute_class_weights(
        splits["train"],
        class_values,
        int(params["ignore_index"]),
        training_cfg["class_weight_strategy"],
    )
    if weights is not None:
        weights = weights.to(device)
    loss_fn = CombinedCrossEntropyDice(
        num_classes=len(class_values),
        ignore_index=int(params["ignore_index"]),
        class_weights=weights,
        cross_entropy_weight=float(training_cfg["cross_entropy_weight"]),
        dice_weight=float(training_cfg["dice_weight"]),
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

    best_score = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, Any]] = []
    best_metric_name = training_cfg["save_best_metric"]

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
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
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(training_cfg["early_stopping_patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = output_path(config, "model_filename")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_values": class_values,
            "params": params,
        },
        model_path,
    )
    test_result = evaluate(model, test_loader, loss_fn, device, len(class_values))
    return model, {
        "device": str(device),
        "class_values": class_values,
        "class_weights": weights.detach().cpu().tolist() if weights is not None else [],
        "best_score": best_score,
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

    features = rgb.astype(np.float32) / 255.0
    window_size = int(inference_cfg["window_size"])
    overlap = int(inference_cfg["overlap"])
    stride = max(1, window_size - overlap)
    batch_size = int(inference_cfg["batch_size"])
    height, width = features.shape[1:]
    prob_sum = np.zeros((len(class_values), height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    pending: List[Tuple[int, int, np.ndarray]] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.from_numpy(np.stack([item[2] for item in pending])).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).detach().cpu().numpy()
        for (y, x, _tile), prob in zip(pending, probs):
            y1 = min(y + window_size, height)
            x1 = min(x + window_size, width)
            prob_sum[:, y:y1, x:x1] += prob[:, : y1 - y, : x1 - x]
            weight_sum[y:y1, x:x1] += 1.0
        pending.clear()

    for y in window_starts(height, window_size, stride):
        for x in window_starts(width, window_size, stride):
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
