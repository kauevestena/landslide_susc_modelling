"""Model training utilities for landslide susceptibility segmentation."""

import os
import json
import random
from typing import Dict, Optional, Tuple, List

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

from src.metrics import select_optimal_thresholds
from src.visualize import generate_all_plots

IGNORE_INDEX = 255


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module (SAM) for focusing on important spatial regions.

    Computes attention weights across spatial dimensions to emphasize
    informative regions for landslide detection. Particularly useful for
    Class 1 (medium-risk) boundary detection.

    Architecture:
    - Channel pooling (max + avg) → Conv 7×7 → Sigmoid
    - Output: spatial attention map [H, W]
    """

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttentionModule, self).__init__()

        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # Convolutional layer to learn spatial attention
        self.conv = nn.Conv2d(
            in_channels=2,  # Max-pool + Avg-pool features
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]

        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Channel-wise pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Concatenate pooled features
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # [B, 2, H, W]

        # Compute attention weights
        attention = self.conv(pooled)  # [B, 1, H, W]
        attention = self.sigmoid(attention)  # [B, 1, H, W]

        # Apply attention
        return x * attention


class UnetWithAttention(nn.Module):
    """
    U-Net wrapper with Spatial Attention Module.

    Adds spatial attention to the deepest encoder features before decoding.
    Helps focus on important terrain features for landslide detection.
    """

    def __init__(self, base_model: nn.Module):
        super(UnetWithAttention, self).__init__()
        self.base_model = base_model
        self.attention = SpatialAttentionModule(kernel_size=7)

        # Copy segmentation_head attribute for compatibility
        self.segmentation_head = base_model.segmentation_head

    def forward(self, x):
        # Get encoder features
        features = self.base_model.encoder(x)

        # Apply spatial attention to the deepest feature map (bottleneck)
        # features is a list: [stage0, stage1, stage2, stage3, stage4, bottleneck]
        attended_features = list(features)
        attended_features[-1] = self.attention(features[-1])

        # Pass attended features to decoder (as a list, not unpacked)
        decoder_output = self.base_model.decoder(attended_features)

        # Apply segmentation head
        output = self.base_model.segmentation_head(decoder_output)
        return output


class LandslideDataset(Dataset):
    """Dataset that loads pre-tiled feature stacks and labels (hard or soft)."""

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

        # Detect if labels are soft (3D) or hard (2D)
        sample_label_path = os.path.join(
            self.labels_dir, os.path.basename(self.tile_files[0])
        )
        sample_label = np.load(sample_label_path)
        self.use_soft_labels = sample_label.ndim == 3

        if self.use_soft_labels:
            print(
                f"LandslideDataset ({split}): Using SOFT labels (shape: {sample_label.shape})"
            )
        else:
            print(
                f"LandslideDataset ({split}): Using HARD labels (shape: {sample_label.shape})"
            )

    def __len__(self) -> int:
        return len(self.tile_files)

    def __getitem__(self, idx: int):
        tile_path = self.tile_files[idx]
        label_path = os.path.join(self.labels_dir, os.path.basename(tile_path))
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found for tile: {tile_path}")

        tile = np.load(tile_path).astype(np.float32)

        if self.use_soft_labels:
            # Soft labels: shape (num_classes, H, W)
            label = np.load(label_path).astype(np.float32)
        else:
            # Hard labels: shape (H, W)
            label = np.load(label_path).astype(np.int16)

        tile, label = self._apply_augmentations(tile, label)

        tile_tensor = torch.from_numpy(tile.copy())

        if self.use_soft_labels:
            # Soft labels stay as float32
            label_tensor = torch.from_numpy(label.copy())
        else:
            # Hard labels converted to int64 for CrossEntropyLoss
            label_tensor = torch.from_numpy(label.astype(np.int64))

        return tile_tensor, label_tensor

    def _apply_augmentations(
        self, tile: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.augment_config
        if self.split != "train" or not cfg:
            return tile, label

        # Determine if label is soft (3D) or hard (2D)
        is_soft = label.ndim == 3

        if cfg.get("flip_prob", 0) > 0 and np.random.rand() < cfg["flip_prob"]:
            tile = tile[:, :, ::-1]
            if is_soft:
                label = label[:, :, ::-1]
            else:
                label = label[:, ::-1]

        if cfg.get("flip_prob", 0) > 0 and np.random.rand() < cfg["flip_prob"]:
            tile = tile[:, ::-1, :]
            if is_soft:
                label = label[:, ::-1, :]
            else:
                label = label[::-1, :]

        if cfg.get("rotate90", True) and np.random.rand() < 0.5:
            k = np.random.randint(0, 4)
            tile = np.rot90(tile, k=k, axes=(1, 2))
            if is_soft:
                label = np.rot90(label, k=k, axes=(1, 2))
            else:
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


class SoftDiceCrossEntropyLoss(nn.Module):
    """
    Loss function for soft (probabilistic) labels in multi-class segmentation.

    Uses KL Divergence instead of CrossEntropy and adapts Dice loss for soft targets.
    Handles both soft labels (num_classes, H, W) and masked pixels (all zeros).
    """

    def __init__(
        self,
        num_classes: int,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = (
            ignore_index  # Not directly used, but kept for API compatibility
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits from model
            targets: (B, C, H, W) soft probability targets that sum to 1.0 per pixel
                     OR (B, H, W) hard class indices (for backward compatibility)

        Returns:
            Scalar loss value
        """
        # Handle both soft and hard targets
        if targets.ndim == 3:
            # Hard labels: use standard CE + Dice
            ce = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)
            dice = smp.losses.DiceLoss(
                mode="multiclass", from_logits=True, ignore_index=self.ignore_index
            )
            return 0.5 * ce(inputs, targets) + 0.5 * dice(inputs, targets)

        # Soft labels from here on: targets shape (B, C, H, W)
        B, C, H, W = inputs.shape

        # Identify valid pixels (where target probabilities sum to > 0)
        target_sum = targets.sum(dim=1)  # (B, H, W)
        valid_mask = target_sum > 1e-6  # (B, H, W)

        if not valid_mask.any():
            # No valid pixels, return zero loss
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # 1. KL Divergence Loss (replaces CrossEntropy for soft labels)
        log_probs = F.log_softmax(inputs, dim=1)  # (B, C, H, W)

        # Only compute loss on valid pixels
        kl_loss = -(targets * log_probs).sum(dim=1)  # (B, H, W)
        kl_loss = kl_loss[valid_mask].mean()

        # 2. Soft Dice Loss
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)

        dice_loss = 0.0
        smooth = 1e-5

        for c in range(C):
            pred_c = probs[:, c, :, :]  # (B, H, W)
            target_c = targets[:, c, :, :]  # (B, H, W)

            # Only compute on valid pixels
            pred_c_valid = pred_c[valid_mask]
            target_c_valid = target_c[valid_mask]

            intersection = (pred_c_valid * target_c_valid).sum()
            union = pred_c_valid.sum() + target_c_valid.sum()

            dice_score = (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += 1.0 - dice_score

        dice_loss = dice_loss / C

        # Blend losses
        return 0.5 * kl_loss + 0.5 * dice_loss


class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss for handling severe class imbalance.

    Focal Loss down-weights easy examples and focuses on hard negatives,
    which is critical for landslide susceptibility where high-risk areas are rare.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        num_classes: int,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
        focal_weight: float = 0.7,
        dice_weight: float = 0.3,
    ):
        """
        Args:
            num_classes: Number of classes
            alpha: Per-class weight tensor (if None, uses uniform weighting)
            gamma: Focusing parameter (higher = more focus on hard examples)
            ignore_index: Index to ignore in loss computation
            focal_weight: Weight for focal loss component (default: 0.7)
            dice_weight: Weight for dice loss component (default: 0.3)
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits from model
            targets: (B, H, W) hard class indices OR (B, C, H, W) soft labels

        Returns:
            Scalar loss value
        """
        # Handle both soft and hard labels
        if targets.ndim == 3:
            # Hard labels: shape (B, H, W)
            return self._focal_dice_hard(inputs, targets)
        else:
            # Soft labels: shape (B, C, H, W)
            return self._focal_dice_soft(inputs, targets)

    def _focal_dice_hard(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal + Dice loss for hard labels."""
        B, C, H, W = inputs.shape

        # Compute softmax probabilities
        probs = F.softmax(inputs, dim=1)  # (B, C, H, W)
        log_probs = F.log_softmax(inputs, dim=1)

        # Create mask for valid pixels
        mask = targets != self.ignore_index  # (B, H, W)
        if not mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # 1. Focal Loss
        # Gather probabilities of true classes
        targets_one_hot = F.one_hot(
            targets.clamp(0, C - 1), num_classes=C
        )  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # pt = probability of true class
        pt = (probs * targets_one_hot).sum(dim=1)  # (B, H, W)

        # Focal weight: (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma

        # Cross-entropy loss per pixel
        ce_loss = -(targets_one_hot * log_probs).sum(dim=1)  # (B, H, W)

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_weight = (
                self.alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * targets_one_hot
            ).sum(dim=1)
            focal_loss = alpha_weight * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        focal_loss = focal_loss[mask].mean()

        # 2. Dice Loss
        dice_loss = 0.0
        smooth = 1e-5

        for c in range(C):
            pred_c = probs[:, c, :, :]
            target_c = (targets == c).float()

            pred_c_valid = pred_c[mask]
            target_c_valid = target_c[mask]

            intersection = (pred_c_valid * target_c_valid).sum()
            union = pred_c_valid.sum() + target_c_valid.sum()

            dice_score = (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += 1.0 - dice_score

        dice_loss = dice_loss / C

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

    def _focal_dice_soft(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal + Dice loss for soft labels."""
        B, C, H, W = inputs.shape

        # Identify valid pixels
        target_sum = targets.sum(dim=1)
        mask = target_sum > 1e-6

        if not mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        probs = F.softmax(inputs, dim=1)
        log_probs = F.log_softmax(inputs, dim=1)

        # 1. Focal-weighted KL Divergence
        # Average probability of target distribution
        pt = (probs * targets).sum(dim=1)  # (B, H, W)
        focal_weight = (1.0 - pt.clamp(0, 1)) ** self.gamma

        kl_loss = -(targets * log_probs).sum(dim=1)  # (B, H, W)
        focal_kl = (focal_weight * kl_loss)[mask].mean()

        # 2. Soft Dice Loss
        dice_loss = 0.0
        smooth = 1e-5

        for c in range(C):
            pred_c = probs[:, c, :, :]
            target_c = targets[:, c, :, :]

            pred_c_valid = pred_c[mask]
            target_c_valid = target_c[mask]

            intersection = (pred_c_valid * target_c_valid).sum()
            union = pred_c_valid.sum() + target_c_valid.sum()

            dice_score = (2.0 * intersection + smooth) / (union + smooth)
            dice_loss += 1.0 - dice_score

        dice_loss = dice_loss / C

        return self.focal_weight * focal_kl + self.dice_weight * dice_loss


class CORALLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) loss for ordinal regression.

    For 3-class ordinal classification (Low=0, Medium=1, High=2),
    CORAL models cumulative probabilities:
      P(Y > 0) = P(Medium or High)
      P(Y > 1) = P(High)

    This explicitly enforces the ordinal constraint that Low < Medium < High,
    reducing confusion between distant classes (e.g., Low vs High).

    Reference: Cao et al. "Rank Consistent Ordinal Regression for Neural Networks
               with Application to Age Estimation" (2020)
    """

    def __init__(self, num_classes: int = 3, ignore_index: int = IGNORE_INDEX):
        """
        Args:
            num_classes: Number of ordinal classes (default: 3 for Low/Med/High)
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # For 3 classes, we have 2 binary cumulative tasks: P(Y>0), P(Y>1)
        self.num_thresholds = num_classes - 1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) logits from model
            targets: (B, H, W) hard class indices (0, 1, 2)

        Returns:
            Scalar CORAL loss
        """
        B, C, H, W = logits.shape

        # Create mask for valid pixels
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Convert logits to cumulative logits
        # For 3 classes: cumulative_logits[0] = P(Y > 0), cumulative_logits[1] = P(Y > 1)
        cumulative_logits = self._get_cumulative_logits(
            logits
        )  # (B, num_thresholds, H, W)

        # Create cumulative labels
        # targets: [0, 1, 2] -> cumulative: [[0, 0], [1, 0], [1, 1]]
        cumulative_labels = self._get_cumulative_labels(
            targets
        )  # (B, num_thresholds, H, W)

        # Compute binary cross-entropy for each cumulative task
        loss = 0.0
        for k in range(self.num_thresholds):
            logits_k = cumulative_logits[:, k, :, :]  # (B, H, W)
            labels_k = cumulative_labels[:, k, :, :].float()  # (B, H, W)

            # Binary cross-entropy with logits
            bce = F.binary_cross_entropy_with_logits(
                logits_k[mask], labels_k[mask], reduction="mean"
            )
            loss += bce

        # Average over all cumulative tasks
        return loss / self.num_thresholds

    def _get_cumulative_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert class logits to cumulative logits.

        For 3 classes [Low, Med, High]:
          cumulative_logits[0] = log(P(Med) + P(High)) / P(Low)
                                = log(1 - P(Low))
          cumulative_logits[1] = log(P(High)) / (P(Low) + P(Med))

        Simplified approach: Use logsumexp trick
        """
        B, C, H, W = logits.shape
        cumulative = []

        for k in range(self.num_thresholds):
            # P(Y > k) = sum of probabilities for classes > k
            # Using logsumexp for numerical stability
            logits_greater = logits[:, k + 1 :, :, :]  # Classes > k
            if logits_greater.shape[1] > 0:
                # log(sum(exp(logits))) for classes > k
                cum_logit = torch.logsumexp(logits_greater, dim=1, keepdim=True)
                cumulative.append(cum_logit)

        return torch.cat(cumulative, dim=1)  # (B, num_thresholds, H, W)

    def _get_cumulative_labels(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Convert ordinal class labels to cumulative labels.

        For targets [0, 1, 2]:
          Class 0 (Low):    [0, 0] (not > 0, not > 1)
          Class 1 (Medium): [1, 0] (yes > 0, not > 1)
          Class 2 (High):   [1, 1] (yes > 0, yes > 1)
        """
        B, H, W = targets.shape
        cumulative = torch.zeros(
            (B, self.num_thresholds, H, W), dtype=torch.long, device=targets.device
        )

        for k in range(self.num_thresholds):
            # Label is 1 if target > k, else 0
            cumulative[:, k, :, :] = (targets > k).long()

        return cumulative


class CombinedOrdinalLoss(nn.Module):
    """
    Combined loss: FocalDice + CORAL for ordinal 3-class segmentation.

    FocalDice handles class imbalance and spatial consistency.
    CORAL enforces ordinal relationships (Low < Medium < High).

    This combination addresses both the severe class imbalance in landslide
    susceptibility (Low: 91%, Medium: 2.6%, High: 5.7%) and the ordinal
    nature of risk levels.
    """

    def __init__(
        self,
        num_classes: int = 3,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
        focal_weight: float = 0.7,
        dice_weight: float = 0.3,
        coral_weight: float = 0.3,
    ):
        """
        Args:
            num_classes: Number of classes (3 for Low/Med/High)
            alpha: Per-class weights for focal loss
            gamma: Focal loss focusing parameter
            ignore_index: Index to ignore
            focal_weight: Weight for focal component (default: 0.7)
            dice_weight: Weight for dice component (default: 0.3)
            coral_weight: Weight for CORAL component (default: 0.3)
        """
        super().__init__()
        self.focal_dice = FocalDiceLoss(
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            ignore_index=ignore_index,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
        )
        self.coral = CORALLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.coral_weight = coral_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C, H, W) logits
            targets: (B, H, W) hard labels or (B, C, H, W) soft labels

        Returns:
            Combined loss value
        """
        # FocalDice handles both hard and soft labels
        focal_dice_loss = self.focal_dice(inputs, targets)

        # CORAL requires hard labels
        if targets.ndim == 4:
            # Convert soft labels to hard by argmax
            targets_hard = torch.argmax(targets, dim=1)
        else:
            targets_hard = targets

        coral_loss = self.coral(inputs, targets_hard)

        # Combined loss: FocalDice (base) + CORAL (ordinal constraint)
        total_loss = focal_dice_loss + self.coral_weight * coral_loss

        return total_loss


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


def _collect_logits_and_labels(
    model: nn.Module, loader: Optional[DataLoader], device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Gather logits and labels (ignoring 255) for calibration."""
    if loader is None or len(loader) == 0:
        return None, None
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for tiles, labels in loader:
            tiles = tiles.to(device)
            labels = labels.to(device)
            outputs = model(tiles)

            # Handle both soft and hard labels
            if labels.ndim == 4:
                # Soft labels: (B, C, H, W)
                # Valid pixels are where probabilities sum > 0
                target_sum = labels.sum(dim=1)  # (B, H, W)
                mask = target_sum > 1e-6
                if not mask.any():
                    continue
                # Convert soft labels to hard by taking argmax
                labels_hard = labels.argmax(dim=1)  # (B, H, W)
                selected_logits = outputs.permute(0, 2, 3, 1)[mask].reshape(
                    -1, outputs.shape[1]
                )
                selected_labels = labels_hard[mask].reshape(-1)
            else:
                # Hard labels: (B, H, W)
                mask = labels != IGNORE_INDEX
                if not mask.any():
                    continue
                selected_logits = outputs.permute(0, 2, 3, 1)[mask].reshape(
                    -1, outputs.shape[1]
                )
                selected_labels = labels[mask].reshape(-1)

            logits_list.append(selected_logits.cpu())
            labels_list.append(selected_labels.cpu())
    if not logits_list:
        return None, None
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return logits, labels


def _optimize_temperature(
    logits: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    max_iter: int = 50,
) -> Tuple[float, float, float]:
    """Fit temperature scaling to reduce negative log likelihood on validation data."""
    if logits is None or labels is None or logits.numel() == 0:
        return 1.0, float("nan"), float("nan")

    device = logits.device
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=max_iter)

    logits = logits.to(device)
    labels = labels.to(device)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(torch.exp(log_temperature).item())
    temperature = max(1.0, min(temperature, 1e2))

    with torch.no_grad():
        nll_before = float(F.cross_entropy(logits, labels).item())
        nll_after = float(F.cross_entropy(logits / temperature, labels).item())

    return temperature, nll_before, nll_after


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

            # Handle both soft and hard labels
            if labels.ndim == 4:
                # Soft labels: (B, C, H, W)
                # Convert to hard labels by taking argmax, but only on valid pixels
                target_sum = labels.sum(dim=1)  # (B, H, W)
                mask = target_sum > 1e-6  # Valid pixels
                if mask.sum() == 0:
                    continue
                labels_hard = labels.argmax(dim=1)  # (B, H, W)
                labels_valid = labels_hard[mask]
                preds_valid = preds[mask]
            else:
                # Hard labels: (B, H, W)
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
    temperature_path = os.path.join(experiments_dir, "temperature_scaling.json")

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
            "temperature_path": (
                temperature_path if os.path.exists(temperature_path) else None
            ),
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

    # V2.5: Add spatial attention if enabled
    use_attention = config["model"].get("attention", False)
    if use_attention:
        print("Wrapping model with Spatial Attention Module...")
        model = UnetWithAttention(model)

    dropout_prob = config["model"].get("dropout_prob", 0.0)
    if dropout_prob and dropout_prob > 0:
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=dropout_prob), model.segmentation_head
        )

    model = model.to(device)

    # Use config class weights if provided, otherwise compute from data
    config_weights = training_cfg.get("class_weights", None)
    if config_weights is not None:
        print(f"Using configured class weights: {config_weights}")
        class_weights = torch.tensor(config_weights, dtype=torch.float32).to(device)
    else:
        print("Computing class weights from dataset summary...")
        class_weights = compute_class_weights(
            os.path.join(splits_dir, "dataset_summary.json"), num_classes
        )
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"Computed class weights: {class_weights.cpu().tolist()}")

    # Detect if we're using soft labels
    use_soft_labels = train_dataset.use_soft_labels

    # Choose loss function based on config
    use_focal_loss = training_cfg.get("use_focal_loss", True)
    use_ordinal_loss = training_cfg.get("use_ordinal_loss", False)
    focal_gamma = training_cfg.get("focal_gamma", 2.0)
    coral_weight = training_cfg.get("coral_weight", 0.3)

    if use_focal_loss and use_ordinal_loss:
        print(f"Training with CombinedOrdinalLoss (FocalDice + CORAL)")
        print(f"  Focal gamma: {focal_gamma}, CORAL weight: {coral_weight}")
        print(
            f"  Class weights (alpha): {class_weights.cpu().tolist() if class_weights is not None else 'None'}"
        )
        loss_fn = CombinedOrdinalLoss(
            num_classes=num_classes,
            alpha=class_weights,
            gamma=focal_gamma,
            ignore_index=IGNORE_INDEX,
            focal_weight=0.7,
            dice_weight=0.3,
            coral_weight=coral_weight,
        )
    elif use_focal_loss:
        print(
            f"Training with FocalDiceLoss (gamma={focal_gamma}, soft_labels={use_soft_labels})"
        )
        print(
            f"  Class weights (alpha): {class_weights.cpu().tolist() if class_weights is not None else 'None'}"
        )
        loss_fn = FocalDiceLoss(
            num_classes=num_classes,
            alpha=class_weights,
            gamma=focal_gamma,
            ignore_index=IGNORE_INDEX,
            focal_weight=0.7,
            dice_weight=0.3,
        )
    elif use_soft_labels:
        print("Training with SoftDiceCrossEntropyLoss (soft labels)")
        loss_fn = SoftDiceCrossEntropyLoss(
            num_classes, weight=class_weights, ignore_index=IGNORE_INDEX
        )
    else:
        print("Training with DiceCrossEntropyLoss (hard labels)")
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
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp)
        if device.type == "cuda"
        else torch.amp.GradScaler("cpu", enabled=use_amp)
    )

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

    temperature_metrics: Optional[Dict[str, float]] = None
    if val_loader is not None and len(val_loader) > 0:
        val_logits, val_label_tensor = _collect_logits_and_labels(
            model, val_loader, device
        )
        if val_logits is not None and val_label_tensor is not None:
            temperature_value, nll_before, nll_after = _optimize_temperature(
                val_logits, val_label_tensor
            )
            temperature_metrics = {
                "temperature": temperature_value,
                "nll_before": nll_before,
                "nll_after": nll_after,
                "num_samples": int(val_label_tensor.numel()),
            }
            with open(temperature_path, "w") as f:
                json.dump(temperature_metrics, f, indent=2)

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
        "temperature_calibration": temperature_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(training_report, f, indent=2)

    temperature_artifact = None
    if temperature_metrics:
        temperature_artifact = temperature_path
    elif os.path.exists(temperature_path):
        temperature_artifact = temperature_path

    return {
        "model_path": best_model_path,
        "calibrator_path": calibrator_path,
        "metrics_path": metrics_path,
        "channel_metadata_path": train_artifacts.metadata_path,
        "normalization_stats_path": train_artifacts.normalization_stats_path,
        "temperature_path": temperature_artifact,
    }


if __name__ == "__main__":
    raise RuntimeError("Run main_pipeline.py to execute the end-to-end workflow.")
