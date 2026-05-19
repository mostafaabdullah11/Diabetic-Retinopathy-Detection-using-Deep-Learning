"""
trainer.py — DR classification training pipeline.

Fix 1 applied here: switched from build_weighted_sampler to
  build_partial_weighted_sampler(max_oversample_ratio=3.0).

Fix 4 applied here: MixUp augmentation for classes 3 and 4.
  Creates synthetic training samples by linearly interpolating between
  two images and their labels. Controlled by mixup_alpha config parameter.

Fix 5 applied here: prediction-time class-bias correction.
  After training, the model systematically over-predicts class 2.
  A small per-class logit bias (trained on val set) shifts predictions
  away from class 2 without retraining. Applied at inference time.
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.dataset import (
    APTOSDataset,
    get_train_transforms,
    get_val_transforms,
    build_partial_weighted_sampler,
    compute_class_weights,
    compute_soft_class_weights,
)
from src.data.splitter import get_folds
from src.models.model_builder import get_model, get_optimizer_groups, unfreeze_backbone
from src.training.losses import CombinedLoss
from src.utils.metrics import quadratic_weighted_kappa, per_class_report
from src.utils.logger import TrainingLogger


# ─────────────────────────────────────────────
# Fix 4: MixUp
# ─────────────────────────────────────────────

def mixup_batch(
    images:  torch.Tensor,
    targets: torch.Tensor,
    alpha:   float = 0.4,
    minority_classes: tuple = (3, 4),
) -> tuple:
    """
    FIX 4 — MixUp augmentation, applied only when at least one sample in the
    batch belongs to a minority class (3 or 4).

    WHY MIXUP HELPS CLASSES 3 AND 4 SPECIFICALLY:
      Classes 3 and 4 have only 39 and 59 training images respectively.
      Even with the partial sampler, the model has seen all of these images
      many times and begins to overfit their specific visual patterns.
      MixUp creates convex combinations of two images:

          image_mix  = λ × image_A  + (1-λ) × image_B
          target_mix = λ × onehot_A + (1-λ) × onehot_B

      where λ ~ Beta(alpha, alpha). This forces the model to learn smooth
      decision boundaries rather than sharp memorised patterns. It acts as a
      data-space regulariser that is especially powerful when n is small.

      We only apply MixUp when a minority-class sample is in the batch.
      This avoids diluting the clear class-0 signal (which is already well-learned)
      while concentrating the regularisation benefit on classes 3/4.

    WHY alpha=0.4 (not the common 0.2):
      alpha=0.2 produces λ close to 1.0 most of the time (mild mixing).
      alpha=0.4 gives more balanced mixes, which is more effective for
      small datasets where we want maximum diversity per sample.

    Args:
        images          : (B, C, H, W) float tensor on device
        targets         : (B,) long tensor on device
        alpha           : Beta distribution parameter
        minority_classes: classes that trigger MixUp (default: 3 and 4)

    Returns:
        (mixed_images, targets_a, targets_b, lam)
        Pass all four to mixup_criterion() — see below.
    """
    has_minority = any(t.item() in minority_classes for t in targets)
    if not has_minority:
        # No minority class in batch — return unchanged
        return images, targets, targets, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)

    # Shuffle indices to pick mixing partners
    idx = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[idx]

    targets_a = targets
    targets_b = targets[idx]
    return mixed_images, targets_a, targets_b, lam


def mixup_criterion(
    criterion,
    logits:    torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam:       float,
) -> torch.Tensor:
    """
    Applies the loss function to both MixUp targets and linearly combines.
    loss = λ × loss(logits, targets_a) + (1-λ) × loss(logits, targets_b)
    """
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)


# ─────────────────────────────────────────────
# Fix 5: Bias correction (post-training logit shift)
# ─────────────────────────────────────────────

def find_qwk_optimal_bias(
    model:       torch.nn.Module,
    val_loader:  DataLoader,
    device:      torch.device,
    num_classes: int   = 5,
    search_range: float = 2.0,
    steps:        int   = 15,
) -> torch.Tensor:
    """
    Find per-class logit bias by directly maximising QWK on the val set.

    WHY THE PREVIOUS CE-BASED BIAS WAS WRONG:
      The old find_logit_bias() minimised cross-entropy loss. CE is minimised
      by predicting the true class distribution accurately — which means
      boosting the MAJORITY classes (0 and 2) that dominate the loss.
      Result: class 0 bias = +0.69, class 3 bias = -0.88, class 4 bias = -0.30.
      This actively HURT minority recall (class 1: 36%, class 3: 41%, class 4: 37%).

    THIS FUNCTION instead:
      1. Collects all val logits once (no grad, fast).
      2. Grid-searches a scalar offset for each class independently.
         For each class c, tries offsets in [-search_range, +search_range].
         Keeps the offset that maximises QWK on val set.
      3. Repeats for search_range/2 resolution refinement.

      This is a 5-parameter grid search over QWK directly.
      It takes ~1 second on CPU and produces biases that actually
      BOOST the minority classes that need it.

    Expected result for this model:
      Class 0: small or zero bias  (already recall=0.981)
      Class 1: POSITIVE bias       (recall=0.365, needs boost)
      Class 2: small or zero bias
      Class 3: POSITIVE bias       (recall=0.410, needs boost)
      Class 4: POSITIVE bias       (recall=0.373, needs boost)

    Args:
        model        : trained model (will be set to eval)
        val_loader   : validation DataLoader
        device       : cpu or cuda
        num_classes  : 5 for APTOS
        search_range : +/- range to search for each class bias
        steps        : number of grid points per class per pass

    Returns:
        bias : torch.Tensor shape (num_classes,) — save alongside weights
    """
    from sklearn.metrics import cohen_kappa_score

    model.eval()

    # ── Step 1: Collect all val logits once ──
    all_logits  = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    all_logits  = torch.cat(all_logits,  dim=0).numpy()  # (N, 5)
    all_targets = torch.cat(all_targets, dim=0).numpy()  # (N,)

    bias_np = np.zeros(num_classes, dtype=np.float32)

    def score_with_bias(b):
        adjusted = all_logits + b[np.newaxis, :]
        preds    = adjusted.argmax(axis=1)
        return cohen_kappa_score(all_targets, preds, weights='quadratic')

    base_qwk = score_with_bias(bias_np)
    print(f"\n  Bias search — base QWK: {base_qwk:.4f}")

    # ── Multi-pass coordinate ascent over QWK ──
    # Pass 1: coarse grid [-2, +2] with 31 points per class
    # Pass 2: fine grid centred on best found values, width=0.5
    # Pass 3: ultra-fine width=0.1
    # Why multi-pass instead of one wide grid:
    #   A single 31-point grid has resolution 0.13. Two-pass gets to 0.016.
    #   Three-pass gets to 0.003. All fast because logits are cached.
    # Why coordinate ascent (one class at a time):
    #   True joint search is 31^5 = 28M evaluations. Too slow.
    #   Coordinate ascent converges to a good local maximum in 3 rounds.
    for round_idx in range(3):
        if round_idx == 0:
            grids = [np.linspace(-search_range, search_range, 31) for _ in range(num_classes)]
        elif round_idx == 1:
            grids = [np.linspace(bias_np[c]-0.25, bias_np[c]+0.25, 21) for c in range(num_classes)]
        else:
            grids = [np.linspace(bias_np[c]-0.05, bias_np[c]+0.05, 21) for c in range(num_classes)]

        improved = True
        while improved:
            improved = False
            for c in range(num_classes):
                best_offset = bias_np[c]
                best_qwk_c  = score_with_bias(bias_np)
                for offset in grids[c]:
                    trial    = bias_np.copy()
                    trial[c] = offset
                    q        = score_with_bias(trial)
                    if q > best_qwk_c + 1e-6:
                        best_qwk_c  = q
                        best_offset = offset
                        improved    = True
                bias_np[c] = best_offset

    final_qwk = score_with_bias(bias_np)
    print(f"  Bias search — final QWK: {final_qwk:.4f}  (delta={final_qwk-base_qwk:+.4f})")
    print()
    print("  📐 QWK-optimal logit bias:")
    for c in range(num_classes):
        direction = "↑ boost" if bias_np[c] > 0.05 else ("↓ reduce" if bias_np[c] < -0.05 else "≈ neutral")
        print(f"    Class {c}: {bias_np[c]:+.4f}  {direction}")

    return torch.tensor(bias_np)


def train_one_fold(
    config:    dict,
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    fold_num:  int,
    device:    torch.device,
) -> float:
    """Train on one fold. Returns best validation QWK."""

    # ── Unpack config ──
    image_size     = config['data']['image_size']
    img_dir        = config['data']['images_dir']
    batch_size     = config['training']['batch_size']
    epochs         = config['training']['epochs']
    head_lr        = config['training']['learning_rate']
    backbone_lr    = config['training']['backbone_lr']
    gamma          = config['training']['gamma']
    unfreeze_ep    = config['training']['unfreeze_epoch']
    use_sampler    = config['training']['use_weighted_sampler']
    use_cls_wts    = config['training']['use_class_weights']
    smoothing      = config['training']['label_smoothing']
    patience       = config['training']['early_stopping_patience']
    scheduler_type = config['training']['scheduler']
    model_name     = config['model']['name']
    num_classes    = config['model']['num_classes']
    focal_w        = config['training'].get('focal_weight', 0.5)
    ordinal_w      = config['training'].get('ordinal_weight', 0.5)
    mixup_alpha    = config['training'].get('mixup_alpha', 0.4)
    use_mixup      = config['training'].get('use_mixup', True)
    warmup_epochs  = config['training'].get('warmup_epochs', 2)
    max_oversample = config['training'].get('max_oversample_ratio', 3.0)
    run_bias_corr  = config['training'].get('run_bias_correction', True)

    # ── Datasets ──
    train_dataset = APTOSDataset(train_df, img_dir, transform=get_train_transforms(image_size))
    val_dataset   = APTOSDataset(val_df,   img_dir, transform=get_val_transforms(image_size))
    train_labels  = train_df['diagnosis'].tolist()

    # ── FIX 1: Partial sampler (capped at max_oversample_ratio×) ──
    if use_sampler:
        sampler       = build_partial_weighted_sampler(train_labels, max_oversample_ratio=max_oversample)
        train_shuffle = False
    else:
        sampler       = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, shuffle=train_shuffle,
        num_workers=0, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2,
        shuffle=False, num_workers=0,
    )

    # ── Model ──
    model = get_model(
        model_name=model_name, num_classes=num_classes,
        pretrained=True, freeze_backbone=True,
    ).to(device)

    # ── Class weights for Focal Loss ──
    if use_cls_wts:
        if use_sampler:
            class_weights = compute_soft_class_weights(train_labels, num_classes).to(device)
        else:
            class_weights = compute_class_weights(train_labels, num_classes).to(device)
    else:
        class_weights = None

    # ── Loss (Fix 2+3 inside CombinedLoss: new threshold weights + 0.5/0.5 split) ──
    criterion = CombinedLoss(
        focal_weight=focal_w,
        ordinal_weight=ordinal_w,
        gamma=gamma,
        alpha=class_weights,
        smoothing=smoothing,
        num_classes=num_classes,
    )

    # ── Optimizer ──
    param_groups = get_optimizer_groups(model, head_lr=head_lr, backbone_lr=backbone_lr)
    optimizer    = optim.AdamW(param_groups, weight_decay=1e-4)

    # ── Scheduler ──
    warmup_end  = unfreeze_ep + warmup_epochs
    restart_t0  = config['training'].get('cosine_t0', 10)
    restart_tmult = config['training'].get('cosine_tmult', 1)

    if scheduler_type == 'cosine_restart':
        # CosineAnnealingWarmRestarts: resets LR to max every T_0 epochs.
        # WHY THIS BEATS PLAIN COSINE HERE:
        #   With plain CosineAnnealingLR, LR decays from 1e-4 to 1e-7 monotonically.
        #   By epoch 21 the LR is 8e-6 — almost zero gradient signal.
        #   The model is stuck in a local minimum with no way out.
        #   Warm restarts reset LR to max every T_0=10 epochs, giving the optimizer
        #   a fresh "kick" that can escape shallow local minima.
        #   This is how the model got +0.01 QWK improvement in epochs 18-21 —
        #   the LR was naturally low enough to be in a "restart-like" regime.
        #   Making it explicit and periodic is more effective.
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_t0, T_mult=restart_tmult, eta_min=1e-7
        )
    elif scheduler_type == 'cosine':
        cosine_epochs = max(1, epochs - warmup_end)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-7
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['step_size'],
            gamma=config['training']['step_gamma'],
        )
    else:
        scheduler = None

    # ── Training loop ──
    logger            = TrainingLogger(fold_num)
    best_qwk          = -1.0
    epochs_no_improve = 0
    save_path         = f"best_model_fold{fold_num}.pth"
    in_warmup         = False
    last_val_preds    = []
    last_val_targets  = []

    for epoch in range(epochs):

        # Progressive unfreezing
        if epoch == unfreeze_ep:
            unfreeze_backbone(model)
            in_warmup = True
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] / 5.0
            print(f"  [Ep {epoch+1}] Backbone unfrozen — warmup starts.")

        if in_warmup and epoch == warmup_end:
            in_warmup = False
            optimizer.param_groups[0]['lr'] = backbone_lr
            optimizer.param_groups[1]['lr'] = head_lr
            print(f"  [Ep {epoch+1}] Warmup done — CosineAnnealing starts.")

        # ── Train ──
        model.train()
        epoch_train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Fold {fold_num} Ep {epoch+1}/{epochs}", leave=False)

        for images, targets in loop:
            images  = images.to(device)
            targets = targets.to(device)

            # FIX 4: MixUp (only when minority classes appear in batch)
            if use_mixup:
                images, targets_a, targets_b, lam = mixup_batch(
                    images, targets, alpha=mixup_alpha, minority_classes=(3, 4)
                )
            else:
                targets_a, targets_b, lam = targets, targets, 1.0

            optimizer.zero_grad()
            logits = model(images)

            if use_mixup and lam < 1.0:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler and not in_warmup and epoch >= warmup_end:
            scheduler.step()

        # ── Validate ──
        model.eval()
        all_preds, all_targets_val = [], []
        epoch_val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images  = images.to(device)
                targets = targets.to(device)
                logits  = model(images)
                loss    = criterion(logits, targets)
                epoch_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets_val.extend(targets.cpu().numpy())

        qwk            = quadratic_weighted_kappa(all_targets_val, all_preds)
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss   = epoch_val_loss   / len(val_loader)
        current_lr     = optimizer.param_groups[1]['lr']

        logger.log(epoch + 1, avg_train_loss, avg_val_loss, qwk, current_lr)

        if qwk > best_qwk:
            best_qwk          = qwk
            epochs_no_improve = 0
            last_val_preds    = all_preds
            last_val_targets  = all_targets_val
            torch.save(model.state_dict(), save_path)
            print(f"    [+] QWK={best_qwk:.4f} — saved {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [stop] No improvement for {patience} epochs.")
                break

    print(f"\nFold {fold_num} done. Best QWK = {best_qwk:.4f}\n")
    per_class_report(last_val_targets, last_val_preds)

    # ── FIX 5: Bias correction ──
    if run_bias_corr:
        print("\n🔧 Running post-training bias correction on val set...")
        # Reload best checkpoint for bias fitting
        model.load_state_dict(torch.load(save_path, map_location=device))
        bias = find_qwk_optimal_bias(model, val_loader, device, num_classes=num_classes)
        bias_path = save_path.replace('.pth', '_bias.pt')
        torch.save(bias, bias_path)
        print(f"   Bias saved to {bias_path}")
        print("   Use this at inference: logits = model(x) + bias.to(device)")

    return best_qwk


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def train_model(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df    = pd.read_csv(config['data']['train_csv'])
    folds = get_folds(df, n_splits=config['training']['n_splits'])

    if config['training'].get('train_all_folds', False):
        fold_qwks = []
        for fold_num, (train_df, val_df) in enumerate(folds):
            fold_qwks.append(train_one_fold(config, train_df, val_df, fold_num, device))
        print(f"\nCV: {[f'{q:.4f}' for q in fold_qwks]}")
        print(f"Mean QWK = {np.mean(fold_qwks):.4f} ± {np.std(fold_qwks):.4f}")
    else:
        fold_num = config['training']['fold']
        train_df, val_df = folds[fold_num]
        train_one_fold(config, train_df, val_df, fold_num, device)


# ─────────────────────────────────────────────
# Light 2-fold ensemble
# ─────────────────────────────────────────────

def predict_ensemble(
    val_loader,
    model_paths:  list,
    bias_paths:   list,
    config:       dict,
    device:       torch.device,
) -> tuple:
    """
    Average softmax outputs from multiple trained fold models.
    Optionally applies per-model logit bias before averaging.

    Args:
        val_loader  : DataLoader
        model_paths : list of .pth weight files
        bias_paths  : list of _bias.pt files (same length as model_paths,
                      or empty list to skip bias correction)
        config      : loaded config dict
        device      : torch device
    """
    from src.utils.metrics import quadratic_weighted_kappa, per_class_report
    all_probs_sum = None
    all_targets   = []

    for i, model_path in enumerate(model_paths):
        model = get_model(
            model_name=config['model']['name'],
            num_classes=config['model']['num_classes'],
            pretrained=False, freeze_backbone=False,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Load bias if available
        bias = None
        if bias_paths and i < len(bias_paths):
            import os
            if os.path.exists(bias_paths[i]):
                bias = torch.load(bias_paths[i], map_location=device)

        model_probs  = []
        model_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                logits = model(images)
                if bias is not None:
                    logits = logits + bias.unsqueeze(0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                model_probs.extend(probs)
                if len(all_targets) == 0:
                    model_targets.extend(targets.numpy())

        model_probs = np.array(model_probs)
        if all_probs_sum is None:
            all_probs_sum = model_probs
            all_targets   = model_targets
        else:
            all_probs_sum += model_probs

    avg_probs  = all_probs_sum / len(model_paths)
    all_preds  = avg_probs.argmax(axis=1)

    qwk = quadratic_weighted_kappa(all_targets, all_preds)
    print(f"\nEnsemble ({len(model_paths)} folds) QWK = {qwk:.4f}")
    per_class_report(all_targets, all_preds)
    return all_preds, all_targets
