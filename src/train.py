"""Model training utilities for landslide susceptibility segmentation."""

import os
import json
import random
from typing import Dict, Optional, Tuple, List

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

from src.metrics import select_optimal_thresholds
from src.visualize import generate_all_plots

IGNORE_INDEX = 255


class LandslideDataset(Dataset):
    """Dataset that loads pre-tiled feature stacks and labels."""

    def __init__(
        self,
        tiles_dir: str,
        labels_dir: str,
        split: str,
        augment_config: Optional[Dict] = None,
    ):
        import glob

        self.tiles_dir = os.path.join(tiles_dir, split)
        self.labels_dir = os.path.join(labels_dir, split)
        self.tile_files = sorted(glob.glob(os.path.join(self.tiles_dir, "*.npy")))
        if not self.tile_files:
            raise ValueError(f'No tiles found for split "{split}" in {self.tiles_dir}')
        self.augment_config = augment_config or {}
        self.split = split

        sample = np.load(self.tile_files[0])
        if sample.ndim != 3:
            raise ValueError("Tiles are expected to have shape (C, H, W).")
        self.num_channels = sample.shape[0]

    def __len__(self) -> int:
        return len(self.tile_files)

    def __getitem__(self, idx: int):
        tile_path = self.tile_files[idx]
        label_path = os.path.join(self.labels_dir, os.path.basename(tile_path))
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found for tile: {tile_path}")

        tile = np.load(tile_path).astype(np.float32)
        label = np.load(label_path).astype(np.int16)

        tile, label = self._apply_augmentations(tile, label)

        tile_tensor = torch.from_numpy(tile.copy())
        label_tensor = torch.from_numpy(label.astype(np.int64))
        return tile_tensor, label_tensor

    def _apply_augmentations(
        self, tile: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.augment_config
        if self.split != "train" or not cfg:
            return tile, label

        if cfg.get("flip_prob", 0) > 0 and np.random.rand() < cfg["flip_prob"]:
            tile = tile[:, :, ::-1]
            label = label[:, ::-1]
        if cfg.get("flip_prob", 0) > 0 and np.random.rand() < cfg["flip_prob"]:
            tile = tile[:, ::-1, :]
            label = label[::-1, :]
        if cfg.get("rotate90", True) and np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            tile = np.rot90(tile, k=k, axes=(1, 2))
            label = np.rot90(label, k=k)
        noise_std = cfg.get("noise_std", 0.0)
        if noise_std and noise_std > 0:
            noise = np.random.normal(0.0, noise_std, size=tile.shape).astype(np.float32)
            tile = tile + noise
        return tile, label


class DiceCrossEntropyLoss(nn.Module):
    """Blend cross-entropy and Dice losses for multi-class segmentation."""

    def __init__(
        self,
        num_classes: int,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, ignore_index=ignore_index
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.ce(inputs, targets) + 0.5 * self.dice(inputs, targets)


def compute_class_weights(
    summary_path: str, num_classes: int
) -> Optional[torch.Tensor]:
    """Derive inverse-frequency class weights from the dataset summary."""
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r") as f:
        summary = json.load(f)
    counts = summary.get("class_pixel_counts", {})
    total = 0
    for cls in range(num_classes):
        total += counts.get(str(cls), 0)
    if total == 0:
        return None
    weights: List[float] = []
    for cls in range(num_classes):
        count = counts.get(str(cls), 0)
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics_from_confusion(
    confusion: np.ndarray,
    correct: int,
    total: int,
    prob_list: Optional[List[np.ndarray]] = None,
    label_list: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Calculate aggregate performance metrics from confusion statistics and probabilities."""
    metrics: Dict[str, float] = {}
    metrics["overall_accuracy"] = float(correct / total) if total else float("nan")

    per_class_iou: List[float] = []
    per_class_f1: List[float] = []
    for cls in range(confusion.shape[0]):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        denom_iou = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        if denom_iou > 0:
            per_class_iou.append(tp / denom_iou)
        if denom_f1 > 0:
            per_class_f1.append(2 * tp / denom_f1)
    metrics["macro_iou"] = (
        float(np.mean(per_class_iou)) if per_class_iou else float("nan")
    )
    metrics["macro_f1"] = float(np.mean(per_class_f1)) if per_class_f1 else float("nan")

    if prob_list and label_list and prob_list[0].size > 0:
        probs = np.concatenate(prob_list)
        labels = np.concatenate(label_list)
        if len(np.unique(labels)) > 1:
            metrics["auroc"] = float(roc_auc_score(labels, probs))
            metrics["auprc"] = float(average_precision_score(labels, probs))
        else:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    positive_class: int,
    collect_probs: bool = False,
) -> Tuple[float, Dict[str, float], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """Run evaluation on a dataloader and gather metrics, optionally storing probabilities."""
    if loader is None or len(loader) == 0:
        metrics = {
            "overall_accuracy": float("nan"),
            "macro_iou": float("nan"),
            "macro_f1": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
        return float("nan"), metrics, (None, None)

    model.eval()
    total_loss = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    correct = 0
    total = 0
    prob_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    with torch.no_grad():
        for tiles, labels in loader:
            tiles = tiles.to(device)
            labels = labels.to(device)
            logits = model(tiles)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            mask = labels != IGNORE_INDEX
            if mask.sum() == 0:
                continue
            labels_valid = labels[mask]
            preds_valid = preds[mask]
            labels_np = labels_valid.cpu().numpy()
            preds_np = preds_valid.cpu().numpy()
            for i in range(num_classes):
                for j in range(num_classes):
                    confusion[i, j] += int(np.sum((labels_np == i) & (preds_np == j)))
            correct += int(np.sum(labels_np == preds_np))
            total += labels_np.size

            if collect_probs:
                prob_high = probs[:, positive_class, :, :][mask].detach().cpu().numpy()
                prob_list.append(prob_high)
                label_list.append(
                    (labels_valid == positive_class).cpu().numpy().astype(np.uint8)
                )

    avg_loss = total_loss / max(len(loader), 1)
    metrics = compute_metrics_from_confusion(
        confusion,
        correct,
        total,
        prob_list if collect_probs else None,
        label_list if collect_probs else None,
    )

    if collect_probs and prob_list:
        probs_concat = np.concatenate(prob_list)
        labels_concat = np.concatenate(label_list)
    else:
        probs_concat = None
        labels_concat = None

    return avg_loss, metrics, (probs_concat, labels_concat)


def train_model(
    config: Dict, train_artifacts, force_recreate: bool = False
) -> Dict[str, Optional[str]]:
    """Train the segmentation model and return artifact paths."""
    structure_cfg = config["project_structure"]
    experiments_dir = structure_cfg["experiments_dir"]
    os.makedirs(experiments_dir, exist_ok=True)

    # Check if model already exists
    best_model_path = os.path.join(experiments_dir, "best_model.pth")
    calibrator_path = os.path.join(experiments_dir, "isotonic_calibrator.joblib")
    metrics_path = os.path.join(experiments_dir, "training_metrics.json")

    if os.path.exists(best_model_path) and not force_recreate:
        print(f"[train] Model already exists at {best_model_path}, skipping training")

        # Return existing artifacts with correct keys matching the full training return
        training_artifacts = {
            "model_path": best_model_path,
            "calibrator_path": (
                calibrator_path if os.path.exists(calibrator_path) else None
            ),
            "metrics_path": metrics_path if os.path.exists(metrics_path) else None,
            "channel_metadata_path": train_artifacts.metadata_path,
            "normalization_stats_path": train_artifacts.normalization_stats_path,
        }
        return training_artifacts

    print(f"[train] Training model (force_recreate={force_recreate})")

    seed = config["reproducibility"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training_cfg = config["training"]
    dataset_cfg = config["dataset"]

    tiles_dir = structure_cfg["tiles_dir"]
    labels_dir = structure_cfg["labels_dir"]
    splits_dir = structure_cfg["splits_dir"]

    augment_cfg = training_cfg.get("augmentations", {})
    train_dataset = LandslideDataset(tiles_dir, labels_dir, "train", augment_cfg)
    try:
        val_dataset = LandslideDataset(tiles_dir, labels_dir, "val", None)
    except ValueError:
        val_dataset = None
    try:
        test_dataset = LandslideDataset(tiles_dir, labels_dir, "test", None)
    except ValueError:
        test_dataset = None

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    # Device selection - force CPU if GPU is not compatible
    # GTX 1050 (sm_61) is not supported by PyTorch 2.x which requires sm_70+
    use_cuda = config.get("training", {}).get("use_cuda", False)

    if use_cuda and torch.cuda.is_available():
        try:
            # Test if GPU is actually usable with a convolution operation
            test_input = torch.randn(1, 3, 32, 32).cuda()
            test_conv = torch.nn.Conv2d(3, 3, 3).cuda()
            _ = test_conv(test_input)
            device = torch.device("cuda")
            print("[train] Using CUDA device for training")
        except (RuntimeError, AssertionError) as e:
            print(f"[train] CUDA test failed: {str(e)[:100]}...")
            print("[train] Falling back to CPU")
            device = torch.device("cpu")
    else:
        if not use_cuda:
            print("[train] use_cuda=False in config, using CPU")
        else:
            print("[train] CUDA not available, using CPU")
        device = torch.device("cpu")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_cfg["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_cfg["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

    in_channels = train_dataset.num_channels
    num_classes = config["model"]["out_classes"]

    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"].get("encoder_weights", "imagenet"),
        in_channels=in_channels,
        classes=num_classes,
    )

    dropout_prob = config["model"].get("dropout_prob", 0.0)
    if dropout_prob and dropout_prob > 0:
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout_prob), model.segmentation_head
        )

    model = model.to(device)

    class_weights = compute_class_weights(
        os.path.join(splits_dir, "dataset_summary.json"), num_classes
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    loss_fn = DiceCrossEntropyLoss(
        num_classes, weight=class_weights, ignore_index=IGNORE_INDEX
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    scheduler = None
    if training_cfg.get("lr_scheduler", "").lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_cfg["epochs"]
        )

    use_amp = training_cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -float("inf")
    best_epoch = -1
    best_metrics: Optional[Dict[str, float]] = None
    best_calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None
    patience = training_cfg.get("early_stopping_patience", 10)
    epochs_no_improve = 0
    history: List[Dict] = []

    best_model_path = os.path.join(experiments_dir, "best_model.pth")

    for epoch in range(training_cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for tiles, labels in train_loader:
            tiles = tiles.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(tiles)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                if training_cfg.get("gradient_clip_norm"):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), training_cfg["gradient_clip_norm"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(tiles)
                loss = loss_fn(outputs, labels)
                loss.backward()
                if training_cfg.get("gradient_clip_norm"):
                    nn.utils.clip_grad_norm_(
                        model.parameters(), training_cfg["gradient_clip_norm"]
                    )
                optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        val_loss, val_metrics, calibration_data = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            num_classes,
            dataset_cfg.get("positive_class", num_classes - 1),
            collect_probs=True,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
        )

        score = val_metrics.get("auprc")
        if score is None or np.isnan(score):
            score = val_metrics.get("macro_iou", float("-inf"))

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_calibration = calibration_data
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                },
                best_model_path,
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if epochs_no_improve >= patience:
            break

    if best_epoch == -1:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_channels": in_channels,
                "num_classes": num_classes,
            },
            best_model_path,
        )

    calibrator_path = None
    if (
        best_calibration
        and best_calibration[0] is not None
        and best_calibration[0].size > 0
    ):
        y_scores, y_true = best_calibration
        if len(np.unique(y_true)) > 1:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_scores, y_true)
            calibrator_path = os.path.join(
                experiments_dir, "isotonic_calibrator.joblib"
            )
            joblib.dump(
                {
                    "calibrator": calibrator,
                    "positive_class": dataset_cfg.get(
                        "positive_class", num_classes - 1
                    ),
                },
                calibrator_path,
            )

    # Evaluate on test set if available
    test_metrics = None
    test_calibration = None
    if test_loader is not None:
        print("[train] Evaluating on test set...")
        # Load best model for test evaluation
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)

        test_loss, test_metrics, test_calibration = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
            num_classes,
            dataset_cfg.get("positive_class", num_classes - 1),
            collect_probs=True,
        )
        print(f"[train] Test set metrics: {test_metrics}")

    # Select optimal classification thresholds
    print("[train] Selecting optimal classification thresholds...")
    val_probs = best_calibration[0] if best_calibration else None
    val_labels = best_calibration[1] if best_calibration else None
    test_probs = test_calibration[0] if test_calibration else None
    test_labels = test_calibration[1] if test_calibration else None

    threshold_results = select_optimal_thresholds(
        val_probs, val_labels, test_probs, test_labels
    )
    print(
        f"[train] Recommended threshold: {threshold_results['recommended_threshold']:.4f} "
        f"(method: {threshold_results['recommendation_method']})"
    )

    # Generate performance visualizations
    print("[train] Generating performance visualizations...")
    figures_dir = os.path.join(experiments_dir, "figures")
    plot_paths = generate_all_plots(
        val_probs, val_labels, test_probs, test_labels, history, figures_dir
    )

    metrics_path = os.path.join(experiments_dir, "training_metrics.json")
    training_report = {
        "history": history,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "best_metrics": best_metrics,
        "test_metrics": test_metrics,
        "thresholds": threshold_results,
        "plots": plot_paths,
        "model": {
            "encoder": config["model"]["encoder"],
            "in_channels": in_channels,
            "num_classes": num_classes,
            "dropout_prob": config["model"].get("dropout_prob", 0.0),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(training_report, f, indent=2)

    return {
        "model_path": best_model_path,
        "calibrator_path": calibrator_path,
        "metrics_path": metrics_path,
        "channel_metadata_path": train_artifacts.metadata_path,
        "normalization_stats_path": train_artifacts.normalization_stats_path,
    }


if __name__ == "__main__":
    raise RuntimeError("Run main_pipeline.py to execute the end-to-end workflow.")
