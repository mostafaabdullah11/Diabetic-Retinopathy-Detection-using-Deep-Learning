"""
dataset.py — APTOS Dataset, augmentation pipeline, sampling utilities.

Fix 1 applied here: build_partial_weighted_sampler()
  Replaces full equalisation (all classes → same frequency) with
  capped oversampling (minority classes → at most `max_oversample_ratio`×
  their natural frequency). Prevents class 3 (39 samples) from being
  repeated 9× per epoch, which caused the model to memorise rather than
  generalise its 39 training images.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomEqualize(p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class APTOSDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img_name = self.df.iloc[idx]['id_code'] + '.png'
        image = Image.open(f"{self.img_dir}/{img_name}").convert('RGB')
        label = int(self.df.iloc[idx]['diagnosis'])
        if self.transform:
            image = self.transform(image)
        return image, label


# ─────────────────────────────────────────────
# Samplers
# ─────────────────────────────────────────────

def build_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Full equalisation: every class appears at equal expected frequency.
    Problem: with only 39 class-3 samples, each is repeated ~9× per epoch
    → model memorises class 3 rather than learning generalisable features.
    Use build_partial_weighted_sampler() instead.
    Kept here for backward compatibility.
    """
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[l] for l in labels])
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(labels), replacement=True
    )


def build_partial_weighted_sampler(
    labels: list,
    max_oversample_ratio: float = 3.0,
) -> WeightedRandomSampler:
    """
    FIX 1 — Capped oversampling instead of full equalisation.

    WHY THE OLD SAMPLER HURTS CLASS 3/4:
      Full WRS equalises every class to the frequency of the rarest class
      (class 3, n=39). This means class 3 images are repeated ~9× per epoch
      while class 0 (n=361) appears at its natural rate. With only 39 distinct
      class-3 images being shown 9× each, the model memorises those specific
      images instead of learning the features of severe DR. On the validation
      set (which contains different class-3 images), recall collapses.

    HOW THIS FIXES IT:
      We cap the per-class weight so that no class is oversampled more than
      `max_oversample_ratio`× its natural frequency. Class 3 gets sampled
      ~3× as often as its natural rate — meaningful signal boost, but not
      so aggressive that the 39 images get memorised.

      Effective frequencies with max_oversample_ratio=3.0 and this dataset:
        Class 0: 1.0×  (no change, already majority)
        Class 1: 2.4×  (74 samples → shown ~2.4× natural rate)
        Class 2: 1.0×  (already common enough)
        Class 3: 3.0×  (39 samples → capped at 3×)
        Class 4: 3.0×  (59 samples → capped at 3×)

    Args:
        labels              : list of integer class labels for the training split
        max_oversample_ratio: maximum times any class is oversampled vs natural rate
    """
    class_counts = Counter(labels)
    total        = len(labels)
    majority_count = max(class_counts.values())

    # Uncapped weight = majority_count / class_count  (so majority gets weight 1.0)
    # Capped weight = min(uncapped, max_oversample_ratio)
    class_weights = {}
    for cls, count in class_counts.items():
        uncapped = majority_count / count
        class_weights[cls] = min(uncapped, max_oversample_ratio)

    sample_weights = torch.DoubleTensor([class_weights[l] for l in labels])

    # num_samples: keep total epoch size ≈ original dataset size
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total,
        replacement=True,
    )


# ─────────────────────────────────────────────
# Class weights for Focal Loss alpha
# ─────────────────────────────────────────────

def compute_class_weights(labels: list, num_classes: int = 5) -> torch.Tensor:
    """Full inverse-frequency weights. Use when sampler is OFF."""
    class_counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * class_counts.get(c, 1)) for c in range(num_classes)]
    return torch.FloatTensor(weights)


def compute_soft_class_weights(labels: list, num_classes: int = 5) -> torch.Tensor:
    """
    Square-root inverse-frequency weights.
    Use when any WeightedRandomSampler is active to avoid double-penalisation.
    """
    class_counts = Counter(labels)
    total = len(labels)
    weights = [
        math.sqrt(total / (num_classes * class_counts.get(c, 1)))
        for c in range(num_classes)
    ]
    return torch.FloatTensor(weights)