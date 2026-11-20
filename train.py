#!/usr/bin/env python3
"""
train.py

Training script for Text-Full-D: SAM 3 with LoRA, text-prompted, full images.

Each training image is processed with one text prompt per class (six forward
passes per image, one per metal subtype). For each prompt, predicted queries are
matched to the ground-truth instances of that class by Hungarian matching, and
the loss combines presence, score, box, and mask terms (see the article,
Section 2.4).

Usage:
  python train.py --config config.yaml

Training resumes automatically from <checkpoint.save_dir>/latest.pt if it exists.
Use --no-auto-resume to start from scratch.
"""

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from model import build_model, build_processor, LoRALinear
from dataloader import FullImageDataset, collate_fn
from utils import (
    load_config, get_device, set_seed, setup_logging, save_config, EarlyStopping,
)


# ══════════════════════════════════════════════════════════════════════════════
# Mixed-precision helpers
# ══════════════════════════════════════════════════════════════════════════════

def amp_dtype_from_name(name: str):
    name = str(name).lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def autocast_context(device, use_amp: bool, dtype):
    if device.type == "cuda" and use_amp and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def cast_inputs(inputs, device, amp_dtype):
    """Move processor outputs to device with appropriate dtype."""
    out = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
        elif v.dtype in (torch.float32, torch.float64):
            if device.type == "cuda" and amp_dtype in (torch.float16, torch.bfloat16):
                out[k] = v.to(device=device, dtype=amp_dtype)
            else:
                out[k] = v.to(device=device, dtype=torch.float32)
        else:
            out[k] = v.to(device=device)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Box geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1).clamp(0.0, 1.0)


def normalize_boxes_xyxy(boxes_abs, h, w):
    if boxes_abs.numel() == 0:
        return boxes_abs
    b = boxes_abs.clone().float()
    b[:, [0, 2]] /= float(w)
    b[:, [1, 3]] /= float(h)
    return b.clamp(0.0, 1.0)


def box_area(boxes):
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def box_iou(b1, b2):
    a1, a2 = box_area(b1), box_area(b2)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - inter + 1e-8
    return inter / union


def generalized_box_iou(b1, b2):
    iou = box_iou(b1, b2)
    lt_e = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb_e = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    wh_e = (rb_e - lt_e).clamp(min=0)
    area_e = wh_e[..., 0] * wh_e[..., 1] + 1e-8
    a1, a2 = box_area(b1), box_area(b2)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - inter + 1e-8
    return iou - (area_e - union) / area_e


# ══════════════════════════════════════════════════════════════════════════════
# Hungarian matching
# ══════════════════════════════════════════════════════════════════════════════

def hungarian_match(pred_logits, pred_boxes_xyxy, gt_boxes_norm, matcher_cfg):
    if gt_boxes_norm.numel() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    G = gt_boxes_norm.shape[0]
    pred_scores = torch.sigmoid(pred_logits).unsqueeze(1).expand(-1, G)
    cls_cost = 1.0 - pred_scores
    l1_cost = torch.cdist(pred_boxes_xyxy, gt_boxes_norm, p=1)
    giou_cost = 1.0 - generalized_box_iou(pred_boxes_xyxy, gt_boxes_norm)

    C = (float(matcher_cfg.get("cls_cost", 1.0)) * cls_cost
         + float(matcher_cfg.get("l1_cost", 5.0)) * l1_cost
         + float(matcher_cfg.get("giou_cost", 2.0)) * giou_cost)

    rows, cols = linear_sum_assignment(C.detach().cpu().float().numpy())
    return rows.astype(np.int64), cols.astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# Loss computation (one call per image and class prompt)
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(logits, targets):
    probs = torch.sigmoid(logits).flatten(1)
    tgts = targets.flatten(1)
    inter = 2.0 * (probs * tgts).sum(dim=1)
    denom = probs.sum(dim=1) + tgts.sum(dim=1) + 1e-8
    return 1.0 - (inter + 1e-8) / denom


def focal_bce_loss(logits, targets, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    focal_weight = (1.0 - pt) ** gamma
    return (focal_weight * bce).mean()


def prepare_pred_masks(pred_masks):
    pm = pred_masks
    if pm.dim() == 4 and pm.shape[1] == 1:
        pm = pm[:, 0]
    if pm.dim() == 4 and pm.shape[0] == 1:
        pm = pm[0]
    if pm.dim() == 2:
        pm = pm.unsqueeze(0)
    return pm


def compute_prompt_losses(
    outputs, model, gt_boxes_abs, gt_masks,
    image_hw, class_present, class_idx, cfg, device,
):
    """
    Loss for one image and one class prompt.

      - Presence loss: whether the prompted class appears in the image.
      - Score, box, and mask losses: computed on queries matched (Hungarian)
        to the ground-truth instances of the prompted class only.
    """
    h, w = image_hw
    loss_cfg = cfg["loss"]
    matcher_cfg = loss_cfg.get("matcher", {"cls_cost": 1.0, "l1_cost": 5.0, "giou_cost": 2.0})

    # Extract predictions
    pred_logits = model.get_pred_logits(outputs)
    if pred_logits.dim() == 3:
        pred_logits = pred_logits[0, :, 0]
    else:
        pred_logits = pred_logits[0].reshape(-1)

    pred_boxes = model.get_pred_boxes(outputs)[0]
    pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    pred_masks = prepare_pred_masks(model.get_pred_masks(outputs))

    # ── Presence loss (focal BCE) ──
    presence_logits = model.get_presence_logits(outputs)[0].reshape(-1).mean().view(1)
    presence_target = torch.tensor(
        [float(class_present)], device=device, dtype=presence_logits.dtype
    )

    focal_gamma = float(loss_cfg.get("presence_focal_gamma", 2.0))
    if focal_gamma > 0:
        presence_loss = focal_bce_loss(presence_logits, presence_target, gamma=focal_gamma)
    else:
        presence_loss = F.binary_cross_entropy_with_logits(presence_logits, presence_target)

    # ── Score, box, mask losses ──
    score_targets = torch.zeros_like(pred_logits)
    box_loss = torch.tensor(0.0, device=device)
    mask_loss = torch.tensor(0.0, device=device)
    n_matched = 0
    matched_iou_sum = 0.0

    if gt_boxes_abs.numel() > 0:
        gt_boxes_norm = normalize_boxes_xyxy(gt_boxes_abs, h, w).to(device)
        rows, cols = hungarian_match(
            pred_logits, pred_boxes_xyxy, gt_boxes_norm, matcher_cfg
        )

        if len(rows) > 0:
            n_matched = len(rows)
            score_targets[rows] = 1.0

            # Box loss
            pb = pred_boxes_xyxy[rows]
            gb = gt_boxes_norm[cols]
            l1 = F.l1_loss(pb, gb)
            giou = 1.0 - torch.diag(generalized_box_iou(pb, gb)).mean()
            box_loss = l1 + giou

            # Mask loss
            gt_float = gt_masks[cols].float().to(device)
            mh, mw = pred_masks.shape[-2], pred_masks.shape[-1]
            gt_resized = F.interpolate(
                gt_float.unsqueeze(1), size=(mh, mw), mode="nearest"
            ).squeeze(1)
            pm = pred_masks[rows]
            mask_bce = F.binary_cross_entropy_with_logits(pm, gt_resized)
            mask_dice = dice_loss(pm, gt_resized).mean()
            mask_loss = mask_bce + mask_dice

            # Track matched mask IoU for monitoring
            with torch.no_grad():
                pred_bin = (torch.sigmoid(pm) > 0.5).float()
                inter = (pred_bin * gt_resized).sum(dim=(1, 2))
                union = pred_bin.sum(dim=(1, 2)) + gt_resized.sum(dim=(1, 2)) - inter + 1e-8
                matched_iou_sum = float((inter / union).sum().item())

    score_loss = F.binary_cross_entropy_with_logits(pred_logits, score_targets)

    # Total weighted loss
    total = (
        float(loss_cfg.get("presence_weight", 0.5)) * presence_loss
        + float(loss_cfg.get("score_weight", 1.0)) * score_loss
        + float(loss_cfg.get("box_weight", 1.0)) * box_loss
        + float(loss_cfg.get("mask_weight", 1.5)) * mask_loss
    )

    stats = {
        "total": float(total.detach().item()),
        "presence": float(presence_loss.detach().item()),
        "score": float(score_loss.detach().item()),
        "box": float(box_loss.detach().item()),
        "mask": float(mask_loss.detach().item()),
        "has_gt": int(gt_boxes_abs.numel() > 0),
        "class_present": int(bool(class_present)),
        "n_matched": n_matched,
        "matched_iou_sum": matched_iou_sum,
    }

    return total, stats


# ══════════════════════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════════════════════

def all_lora_parameters(model):
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_scheduler(optimizer, total_updates, warmup_updates, base_lr, min_lr):
    min_ratio = float(min_lr) / max(float(base_lr), 1e-12)

    def lr_lambda(step):
        if step < warmup_updates:
            return float(step + 1) / max(1, warmup_updates)
        progress = float(step - warmup_updates) / max(1, total_updates - warmup_updates)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_loader_kwargs(num_workers, pin_memory, persistent_workers, prefetch_factor):
    kwargs = {"num_workers": int(num_workers), "pin_memory": bool(pin_memory)}
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def maybe_enable_memory_savers(model, cfg, logger):
    gcfg = cfg["training"].get("memory", {})
    if not bool(gcfg.get("gradient_checkpointing", True)):
        return
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        try:
            model.backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        except Exception as e:
            logger.warning(
                f"Could not enable gradient checkpointing: "
                f"{type(e).__name__}: {e}"
            )
    else:
        logger.info(
            "Gradient checkpointing not available on this backbone; "
            "continuing without it."
        )
    if hasattr(model.backbone, "config") and hasattr(model.backbone.config, "use_cache"):
        try:
            model.backbone.config.use_cache = False
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Per-class loss statistics (for logging)
# ══════════════════════════════════════════════════════════════════════════════

def init_classwise_stats(class_names):
    return {
        name: {
            "total_sum": 0.0, "presence_sum": 0.0, "score_sum": 0.0,
            "box_sum": 0.0, "mask_sum": 0.0,
            "count": 0, "present_count": 0, "gt_count": 0,
            "n_matched": 0, "matched_iou_sum": 0.0,
        }
        for name in class_names
    }


def update_classwise_stats(classwise_stats, class_name, prompt_stats):
    s = classwise_stats[class_name]
    s["total_sum"] += float(prompt_stats["total"])
    s["presence_sum"] += float(prompt_stats["presence"])
    s["score_sum"] += float(prompt_stats["score"])
    s["box_sum"] += float(prompt_stats["box"])
    s["mask_sum"] += float(prompt_stats["mask"])
    s["count"] += 1
    s["present_count"] += int(prompt_stats["class_present"])
    s["gt_count"] += int(prompt_stats["has_gt"])
    s["n_matched"] += int(prompt_stats.get("n_matched", 0))
    s["matched_iou_sum"] += float(prompt_stats.get("matched_iou_sum", 0.0))


def summarize_classwise_stats(classwise_stats):
    summary = {}
    for class_name, s in classwise_stats.items():
        c = max(1, int(s["count"]))
        nm = max(1, int(s["n_matched"]))
        summary[class_name] = {
            "total": s["total_sum"] / c,
            "presence": s["presence_sum"] / c,
            "score": s["score_sum"] / c,
            "box": s["box_sum"] / c,
            "mask": s["mask_sum"] / c,
            "count": int(s["count"]),
            "present_rate": float(s["present_count"]) / c,
            "gt_rate": float(s["gt_count"]) / c,
            "matched_iou": s["matched_iou_sum"] / nm if s["n_matched"] > 0 else 0.0,
        }
    return summary


def log_classwise_stats(logger, split_name, epoch, summary):
    logger.info(f"{split_name} epoch {epoch}: class-wise mean prompt losses")
    for class_name, s in summary.items():
        logger.info(
            f"  {class_name:<22s} "
            f"total={s['total']:.6f} "
            f"pres={s['presence']:.6f} "
            f"score={s['score']:.6f} "
            f"box={s['box']:.6f} "
            f"mask={s['mask']:.6f} "
            f"mIoU={s['matched_iou']:.4f} "
            f"n={s['count']} "
            f"pres_rate={s['present_rate']:.3f} "
            f"gt_rate={s['gt_rate']:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Epoch loop: one text prompt per class on each full image
# (six forward passes per image, one per class)
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    loader, model, processor, optimizer, scheduler,
    device, cfg, epoch, logger, training=True,
):
    model.train(training)

    use_amp = bool(cfg["training"]["use_amp"])
    amp_dtype = amp_dtype_from_name(cfg["training"].get("amp_dtype", "bf16"))
    grad_accum = int(cfg["training"]["grad_accum_steps"])
    class_names = cfg["dataset"]["class_names"]
    num_classes = len(class_names)

    total_loss = 0.0
    n_samples = 0
    accum_counter = 0
    classwise_stats = init_classwise_stats(class_names)

    # Global accumulators for epoch summary
    total_matched = 0
    total_gt = 0
    total_matched_iou_sum = 0.0

    if training:
        optimizer.zero_grad(set_to_none=True)

    desc = f"{'Train' if training else 'Valid'} E{epoch:02d}"
    iterator = tqdm(loader, desc=desc)
    grad_ctx = torch.enable_grad if training else torch.no_grad

    for batch in iterator:
        batch_loss = torch.tensor(0.0, device=device)
        n_batch = 0

        for sample in batch:
            pil = sample["image"]
            h, w = sample["orig_size"]
            gt_boxes = sample["boxes"]       # [N, 4] xyxy absolute
            gt_masks = sample["masks"]       # [N, H, W] uint8
            gt_labels = sample["labels"]     # [N] class indices

            # Build class_present vector
            class_present = torch.zeros(num_classes, dtype=torch.float32)
            for lbl in gt_labels:
                class_present[int(lbl)] = 1.0

            sample_loss = torch.tensor(0.0, device=device)

            # One text prompt per class
            with grad_ctx():
                for class_idx, class_name in enumerate(class_names):
                    # Filter GT to only this class
                    keep = (gt_labels == class_idx)
                    gt_boxes_cls = gt_boxes[keep].to(device)
                    gt_masks_cls = gt_masks[keep].to(device)
                    present = bool(class_present[class_idx].item() > 0)

                    # Single-class text prompt
                    inputs = processor(
                        images=pil, text=class_name, return_tensors="pt"
                    )
                    inputs_dev = cast_inputs(inputs, device, amp_dtype)

                    with autocast_context(device, use_amp, amp_dtype):
                        outputs = model.forward_backbone(
                            **inputs_dev, output_hidden_states=False
                        )

                        prompt_loss, prompt_stats = compute_prompt_losses(
                            outputs=outputs,
                            model=model,
                            gt_boxes_abs=gt_boxes_cls,
                            gt_masks=gt_masks_cls,
                            image_hw=(h, w),
                            class_present=present,
                            class_idx=class_idx,
                            cfg=cfg,
                            device=device,
                        )
                        sample_loss = sample_loss + prompt_loss

                    update_classwise_stats(classwise_stats, class_name, prompt_stats)
                    total_matched += prompt_stats.get("n_matched", 0)
                    total_gt += int(keep.sum().item())
                    total_matched_iou_sum += prompt_stats.get("matched_iou_sum", 0.0)

            batch_loss = batch_loss + sample_loss
            total_loss += float(sample_loss.detach().item())
            n_samples += 1
            n_batch += 1

        if n_batch == 0:
            continue

        batch_loss = batch_loss / float(n_batch)

        if training:
            (batch_loss / grad_accum).backward()
            accum_counter += 1

            if accum_counter >= grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    all_lora_parameters(model),
                    max_norm=float(cfg["training"]["gradient_clip_norm"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

        # Progress bar
        n = max(n_samples, 1)
        nm = max(total_matched, 1)
        avg_iou = total_matched_iou_sum / nm if total_matched > 0 else 0.0
        iterator.set_postfix(
            loss=f"{total_loss / n:.4f}",
            mIoU=f"{avg_iou:.3f}",
        )

    # Flush remaining gradients
    if training and accum_counter > 0:
        torch.nn.utils.clip_grad_norm_(
            all_lora_parameters(model),
            max_norm=float(cfg["training"]["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Log epoch summary
    n = max(n_samples, 1)
    nm = max(total_matched, 1)
    avg_iou = total_matched_iou_sum / nm if total_matched > 0 else 0.0
    mean_loss = total_loss / n
    split_name = "Train" if training else "Valid"

    classwise_summary = summarize_classwise_stats(classwise_stats)

    logger.info(
        f"{split_name} epoch {epoch:02d}: "
        f"loss={mean_loss:.6f} | "
        f"matched_maskIoU={avg_iou:.4f} | "
        f"images={n_samples} | "
        f"matched={total_matched}/{total_gt} GT"
    )
    log_classwise_stats(logger, split_name, epoch, classwise_summary)

    return mean_loss, classwise_summary


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint management
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }, path)


def resolve_resume_path(args, cfg, ckpt_dir: Path):
    if args.resume_from is not None:
        p = Path(args.resume_from)
        return p if p.exists() else None
    cfg_resume = cfg.get("checkpoint", {}).get("resume_from", None)
    if cfg_resume not in [None, "", "null"]:
        p = Path(cfg_resume)
        return p if p.exists() else None
    if args.auto_resume:
        latest = ckpt_dir / "latest.pt"
        if latest.exists():
            return latest
    return None


def load_training_state(resume_path, model, optimizer, scheduler, logger):
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        f"Loaded checkpoint: {resume_path} | "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Loaded optimizer state.")
    if "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            logger.info("Loaded scheduler state.")
        except Exception as e:
            logger.warning(f"Could not load scheduler state: {e}")
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_metric = float(ckpt.get("best_metric", float("inf")))
    return start_epoch, best_metric


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Train SAM 3 with LoRA (Text-Full-D)"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume-from", default=None,
                    help="Path to checkpoint for manual resume")
    ap.add_argument("--no-auto-resume", action="store_false", dest="auto_resume",
                    help="Disable auto-resume from latest.pt")
    ap.set_defaults(auto_resume=True)
    args = ap.parse_args()

    # ── Load config ──
    cfg = load_config(args.config)
    set_seed(
        int(cfg["hardware"].get("seed", 42)),
        deterministic=bool(cfg["hardware"].get("deterministic", False)),
    )
    device = get_device()

    # ── Directories ──
    out_dir = Path(cfg["output_dir"])
    ckpt_dir = Path(cfg["checkpoint"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir)

    # ── Logging ──
    logger = setup_logging(out_dir, log_file="train.log")
    logger.info("=" * 70)
    logger.info("Text-Full-D: SAM 3 with LoRA, one text prompt per class, full images")
    logger.info("=" * 70)

    # Log LoRA configuration
    lora_cfg = cfg["model"]["lora"]
    logger.info(f"  LoRA alpha:        {lora_cfg['alpha']}")
    logger.info(f"  LoRA dropout:      {lora_cfg['dropout']}")
    logger.info(f"  Vision encoder r:  {lora_cfg['vision']['r']}")
    logger.info(f"  DETR encoder r:    {lora_cfg['detr_encoder']['r']}")
    logger.info(f"  DETR decoder r:    {lora_cfg['detr_decoder']['r']}")
    logger.info(f"  Mask decoder r:    {lora_cfg['mask_decoder']['r']}")
    logger.info(f"  Learning rate:     {cfg['training']['learning_rate']['lora']}")
    logger.info("  Prompt mode:       one text prompt per class")
    logger.info("  Image mode:        full image")
    logger.info("=" * 70)

    # ── Build model ──
    processor = build_processor(cfg)
    model = build_model(cfg).to(device)

    # Enable ALL LoRA groups from epoch 1 (single-stage)
    model.configure_trainable_lora(stage=3)
    trainable_count = count_trainable(model)
    logger.info(f"All LoRA groups active | trainable_params={trainable_count:,}")

    maybe_enable_memory_savers(model, cfg, logger)

    # ── Datasets ──
    class_names = cfg["dataset"]["class_names"]
    aug_cfg = cfg.get("augmentation", None)

    train_ds = FullImageDataset(
        dataset_root=cfg["dataset"]["root"],
        split="train",
        class_names=class_names,
        augmentation_cfg=aug_cfg,
    )
    val_ds = FullImageDataset(
        dataset_root=cfg["dataset"]["root"],
        split="valid",
        class_names=class_names,
        augmentation_cfg=None,  # Never augment validation
    )

    logger.info(f"Train images: {len(train_ds)}")
    logger.info(f"Valid images: {len(val_ds)}")

    # ── Data loaders ──
    hw = cfg["hardware"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size_images"]),
        shuffle=True,
        collate_fn=collate_fn,
        **build_loader_kwargs(
            hw["num_workers"], hw["pin_memory"],
            hw["persistent_workers"], hw["prefetch_factor"],
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        **build_loader_kwargs(
            max(1, int(hw["num_workers"]) // 2), hw["pin_memory"],
            False, max(2, int(hw["prefetch_factor"])),
        ),
    )

    # ── Optimizer and scheduler ──
    params = all_lora_parameters(model)
    base_lr = float(cfg["training"]["learning_rate"]["lora"])
    optimizer = AdamW(
        params,
        lr=base_lr,
        weight_decay=float(cfg["training"]["weight_decay"]),
        betas=tuple(cfg["training"]["betas"]),
    )

    num_epochs = int(cfg["training"]["num_epochs"])
    # With sequential prompts, each image produces 6 forward passes,
    # but the scheduler steps per optimizer update (after grad_accum batches)
    steps_per_epoch = math.ceil(
        len(train_loader) / int(cfg["training"]["grad_accum_steps"])
    )
    total_updates = steps_per_epoch * num_epochs
    warmup_updates = steps_per_epoch * int(
        cfg["training"]["lr_scheduler"]["warmup_epochs"]
    )

    scheduler = build_scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        base_lr=base_lr,
        min_lr=float(cfg["training"]["lr_scheduler"]["min_lr"]),
    )

    logger.info(
        f"Optimizer: AdamW lr={base_lr} wd={cfg['training']['weight_decay']} | "
        f"Scheduler: cosine warmup={cfg['training']['lr_scheduler']['warmup_epochs']}ep "
        f"min_lr={cfg['training']['lr_scheduler']['min_lr']}"
    )
    logger.info(
        f"Steps/epoch: {steps_per_epoch} | "
        f"Total updates: {total_updates} | "
        f"Grad accum: {cfg['training']['grad_accum_steps']}"
    )

    # ── Early stopping ──
    early_cfg = cfg["training"].get("early_stopping", {})
    early = EarlyStopping(
        patience=int(early_cfg.get("patience", 10)),
        min_delta=float(early_cfg.get("min_delta", 0.0)),
        mode=str(early_cfg.get("mode", "min")),
        verbose=True,
    )

    # ── Resume ──
    start_epoch = 1
    best_val = float("inf")
    resume_path = resolve_resume_path(args, cfg, ckpt_dir)

    if resume_path is not None:
        logger.info(f"Resuming from: {resume_path}")
        start_epoch, best_val = load_training_state(
            resume_path, model, optimizer, scheduler, logger
        )
        logger.info(f"Resuming at epoch {start_epoch} with best_val={best_val:.6f}")
    else:
        logger.info("Starting fresh from epoch 1.")

    # ══════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════

    current_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            current_epoch = epoch
            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"trainable_params={trainable_count:,}"
            )

            # ── Train ──
            train_loss, train_classwise = run_epoch(
                loader=train_loader,
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                cfg=cfg,
                epoch=epoch,
                logger=logger,
                training=True,
            )

            # ── Validate ──
            val_loss, val_classwise = run_epoch(
                loader=val_loader,
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                cfg=cfg,
                epoch=epoch,
                logger=logger,
                training=False,
            )

            # ── Checkpoint management ──
            if val_loss < best_val:
                best_val = val_loss
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(
                    best_path, model, optimizer, scheduler,
                    epoch, best_val, cfg
                )
                logger.info(f"New best checkpoint: val_loss={best_val:.6f}")

            # Always save latest (for auto-resume)
            save_checkpoint(
                ckpt_dir / "latest.pt", model, optimizer, scheduler,
                epoch, best_val, cfg
            )

            # Periodic checkpoint
            if epoch % int(cfg["training"]["save_every_n_epochs"]) == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                    scheduler, epoch, best_val, cfg
                )

            # Find worst class for reference
            worst_class = (
                max(val_classwise.items(), key=lambda kv: kv[1]["total"])[0]
                if val_classwise else "n/a"
            )

            logger.info(
                f"Epoch {epoch:02d} done | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"best_val={best_val:.6f} | "
                f"worst_class={worst_class}"
            )

            # ── Early stopping ──
            if early(val_loss):
                logger.info("Early stopping triggered.")
                break

    except KeyboardInterrupt:
        interrupt_path = ckpt_dir / "interrupt.pt"
        save_checkpoint(
            interrupt_path, model, optimizer, scheduler,
            current_epoch, best_val, cfg
        )
        logger.info(f"Interrupted — saved: {interrupt_path}")
        raise

    logger.info("Training finished.")


if __name__ == "__main__":
    main()