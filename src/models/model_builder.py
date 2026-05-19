"""
model_builder.py — Model factory with fine-tuning support
Key upgrades vs v1:
  - Added MobileNetV3-Small: fastest on CPU, surprisingly strong
  - Added EfficientNet-B0: best accuracy if you have GPU
  - get_optimizer_with_discriminative_lr(): backbone gets 10x smaller LR than head
  - freeze_backbone() / unfreeze_backbone() helpers for progressive unfreezing
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────

def get_model(
    model_name: str = 'mobilenet_v3_small',
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Build a pretrained classification model with a custom head.

    Supported models:
      - 'resnet18'           : baseline, decent on CPU
      - 'mobilenet_v3_small' : best CPU choice — fast + accurate
      - 'efficientnet_b0'    : best accuracy if GPU available

    Args:
        model_name      : architecture name (see above)
        num_classes     : output classes (5 for APTOS)
        pretrained      : load ImageNet weights
        freeze_backbone : freeze all layers except final head initially

    Returns:
        nn.Module ready for training
    """
    weights_flag = pretrained

    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = _build_head(in_features, num_classes)
        backbone_params = [p for name, p in model.named_parameters() if 'fc' not in name]
        head_params = list(model.fc.parameters())

    elif model_name == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = _build_head(in_features, num_classes)
        backbone_params = [p for name, p in model.named_parameters()
                           if 'classifier.3' not in name]
        head_params = list(model.classifier[3].parameters())

    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = _build_head(in_features, num_classes)
        backbone_params = [p for name, p in model.named_parameters()
                           if 'classifier.1' not in name]
        head_params = list(model.classifier[1].parameters())

    else:
        raise ValueError(f"Unsupported model: '{model_name}'. "
                         f"Choose from: resnet18, mobilenet_v3_small, efficientnet_b0")

    # Store param groups as model attributes for easy access later
    model._backbone_params = backbone_params
    model._head_params = head_params

    if freeze_backbone:
        for param in backbone_params:
            param.requires_grad = False

    return model


def _build_head(in_features: int, num_classes: int) -> nn.Sequential:
    """
    Custom classification head with dropout for regularization.
    Dropout(0.3) reduces overfitting — important for small medical datasets.
    """
    return nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )


# ─────────────────────────────────────────────
# Freeze / Unfreeze Helpers
# ─────────────────────────────────────────────

def freeze_backbone(model: nn.Module):
    """Freeze all backbone layers (only head trains)."""
    for param in model._backbone_params:
        param.requires_grad = False
    print("🔒 Backbone frozen — training head only")


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all layers for full fine-tuning."""
    for param in model._backbone_params:
        param.requires_grad = True
    print("🔓 Backbone unfrozen — full model training")


# ─────────────────────────────────────────────
# Discriminative Learning Rates
# ─────────────────────────────────────────────

def get_optimizer_groups(
    model: nn.Module,
    head_lr: float = 1e-4,
    backbone_lr: float = 1e-5
) -> List[Dict]:
    """
    Returns param groups with different LRs:
      - Backbone (pretrained features): small LR to preserve learned features
      - Head (new classifier):          larger LR to train fast

    Why: If you use the same LR for both, either the backbone overfits
    (high LR) or the head learns too slowly (low LR).

    Args:
        model       : model built with get_model()
        head_lr     : learning rate for classification head
        backbone_lr : learning rate for pretrained backbone (typically head_lr / 10)

    Returns:
        List of dicts suitable for torch.optim.Adam(param_groups, ...)
    """
    return [
        {'params': model._backbone_params, 'lr': backbone_lr},
        {'params': model._head_params,     'lr': head_lr},
    ]
