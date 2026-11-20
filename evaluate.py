#!/usr/bin/env python3
"""
evaluate.py

Evaluation of Text-Full-D (SAM 3 with LoRA) on full images (article, Section 2.5).

For each image:
  1. One text prompt per class (six forward passes). Predictions with a
     confidence score below --score-threshold are discarded.
  2. Predictions from all prompts are pooled and passed through
     class-agnostic mask NMS (--nms-iou-threshold).
  3. Predictions are matched one-to-one to ground-truth objects by Hungarian
     assignment on mask IoU. A match is a true positive (TP) only if its IoU
     is at least 0.50 and the predicted class equals the ground-truth class.
     A correctly located object with the wrong class counts as one false
     positive (FP) and one false negative (FN).

Reported per class and as the mean over classes:
  mIoU@TP    mean IoU of TP matches (classes without any TP are omitted)
  mIoU@all   as mIoU@TP, but every ground-truth object without a TP counts as IoU = 0
  Precision, Recall, F1
  LocRec50   class-agnostic recall (an object counts as found if a prediction
             of any class matches it with IoU >= 0.50)
COCO mask AP (including AP50 = mAP@50) is computed with pycocotools.
Timing: end-to-end time per image (all six prompts, post-processing, NMS, and
metric computation), excluding model loading and saving qualitative images.

Usage:
  python evaluate.py \
    --config config.yaml \
    --checkpoint outputs/text_full_d/checkpoints/best.pt \
    --split test \
    --results-dir outputs/text_full_d/results \
    --score-threshold 0.15 --mask-threshold 0.50 --nms-iou-threshold 0.50
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as maskutil
except ImportError as e:
    raise ImportError("pycocotools required") from e

from model import build_model, build_processor
from utils import load_config, get_device, set_seed


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def cast_inputs(inputs, device):
    out = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
        elif v.dtype in (torch.float32, torch.float64):
            out[k] = v.to(device=device,
                          dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        else:
            out[k] = v.to(device=device)
    return out


def ann_to_mask(ann, h, w):
    seg = ann.get("segmentation", None)
    if not seg:
        return None
    if isinstance(seg, dict):
        rle = maskutil.frPyObjects(seg, h, w) if isinstance(seg.get("counts"), list) else seg
        return maskutil.decode(rle).astype(np.uint8)
    if isinstance(seg, list) and len(seg) > 0:
        return maskutil.decode(maskutil.merge(maskutil.frPyObjects(seg, h, w))).astype(np.uint8)
    return None


def pairwise_mask_iou(pred_masks, gt_masks):
    n, m = len(pred_masks), len(gt_masks)
    iou = np.zeros((n, m), dtype=np.float32)
    for i, pm in enumerate(pred_masks):
        pmb = pm.astype(bool)
        for j, gm in enumerate(gt_masks):
            gmb = gm.astype(bool)
            if pmb.shape != gmb.shape:
                continue
            inter = float((pmb & gmb).sum())
            union = float((pmb | gmb).sum())
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def mask_iou_np(m1, m2):
    m1b, m2b = m1.astype(bool), m2.astype(bool)
    if m1b.shape != m2b.shape:
        return 0.0
    inter = float((m1b & m2b).sum())
    union = float((m1b | m2b).sum())
    return inter / union if union > 0 else 0.0


def cxcywh_to_xyxy_abs(boxes, h, w):
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - 0.5 * bw); y1 = (cy - 0.5 * bh)
    x2 = (cx + 0.5 * bw); y2 = (cy + 0.5 * bh)
    out = torch.stack([x1, y1, x2, y2], dim=-1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, w)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, h)
    return out


def box_iou_xyxy_np(b1, b2):
    xi1 = max(float(b1[0]), float(b2[0]))
    yi1 = max(float(b1[1]), float(b2[1]))
    xi2 = min(float(b1[2]), float(b2[2]))
    yi2 = min(float(b1[3]), float(b2[3]))
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, float(b1[2]) - float(b1[0])) * max(0, float(b1[3]) - float(b1[1]))
    a2 = max(0, float(b2[2]) - float(b2[0])) * max(0, float(b2[3]) - float(b2[1]))
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Class-agnostic mask NMS
# ══════════════════════════════════════════════════════════════════════════════

def class_agnostic_mask_nms(preds, iou_thr=0.5):
    """Suppress overlapping predictions regardless of class."""
    if len(preds) <= 1:
        return preds
    scores = np.array([p["score"] for p in preds])
    order = scores.argsort()[::-1]
    keep = []
    suppressed = set()
    for i in order:
        if int(i) in suppressed:
            continue
        keep.append(int(i))
        for j in order:
            if int(j) in suppressed or int(j) == int(i):
                continue
            iou = mask_iou_np(preds[int(i)]["mask"], preds[int(j)]["mask"])
            if iou > iou_thr:
                suppressed.add(int(j))
    return [preds[k] for k in keep]


# ══════════════════════════════════════════════════════════════════════════════
# Inference with one text prompt (one class) on the full image
# ══════════════════════════════════════════════════════════════════════════════

def collect_prompt_predictions(
    model, processor, device, pil,
    class_name, class_idx, h, w,
    score_thr, mask_thr,
):
    """
    Run one class prompt on the full image.
    Returns a list of predictions, each assigned the prompted class.
    """
    inputs = processor(images=pil, text=class_name, return_tensors="pt")
    inputs_dev = cast_inputs(inputs, device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.forward_backbone(**inputs_dev)
        else:
            outputs = model.forward_backbone(**inputs_dev)

    raw_boxes_abs = cxcywh_to_xyxy_abs(model.get_pred_boxes(outputs)[0], h, w)
    raw_scores = torch.sigmoid(model.get_pred_logits(outputs)[0].reshape(-1))

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=float(score_thr),
        mask_threshold=float(mask_thr),
        target_sizes=[(h, w)],
    )[0]

    preds = []
    masks = results.get("masks", [])
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])

    if len(masks) == 0:
        return preds

    for i in range(len(masks)):
        box = boxes[i].detach().cpu()
        score = float(scores[i].detach().cpu().item()) if len(scores) > 0 else 0.0

        mask_raw = masks[i].detach().cpu().numpy()
        mask_2d = np.squeeze(mask_raw)
        if mask_2d.ndim != 2:
            mask_2d = mask_2d[0]
        if mask_2d.dtype == np.bool_:
            mask = mask_2d.astype(np.uint8)
        else:
            mask = (mask_2d > float(mask_thr)).astype(np.uint8)

        if int(mask.sum()) <= 0:
            continue

        # Raw score of the closest query (stored with each prediction)
        best_q = 0
        best_iou = -1.0
        for q in range(raw_boxes_abs.shape[0]):
            biou = box_iou_xyxy_np(box.tolist(), raw_boxes_abs[q].detach().cpu().tolist())
            if biou > best_iou:
                best_iou = biou
                best_q = q

        preds.append({
            "box": box.numpy().astype(np.float32),
            "mask": mask,
            "score": score,
            "prompt_class": int(class_idx),
            "pred_class": int(class_idx),
            "raw_score": float(raw_scores[best_q].detach().cpu().item()),
        })

    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Qualitative output
# ══════════════════════════════════════════════════════════════════════════════

def get_colors(nc):
    palette = [(255, 80, 80), (255, 180, 60), (80, 220, 120), (80, 160, 255),
               (180, 80, 255), (255, 80, 180), (0, 200, 200), (220, 220, 80)]
    return [palette[i % len(palette)] for i in range(nc)]


def overlay_mask(base, mask, color, alpha=90):
    base = base.convert("RGBA")
    mi = Image.fromarray(mask.astype(np.uint8) * alpha, mode="L")
    ci = Image.new("RGBA", base.size, color + (0,))
    ci.putalpha(mi)
    return Image.alpha_composite(base, ci)


def save_qualitative_image(save_path, pil, preds, gt_items, class_names):
    colors = get_colors(len(class_names))
    w, h = pil.size

    gt_img = pil.convert("RGBA")
    for g in gt_items:
        gt_img = overlay_mask(gt_img, g["mask"], colors[int(g["label"])], alpha=80)

    pred_img = pil.convert("RGBA")
    for p in preds:
        pred_img = overlay_mask(pred_img, p["mask"], colors[int(p["pred_class"])], alpha=80)

    canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
    canvas.paste(gt_img.convert("RGB"), (0, 0))
    canvas.paste(pred_img.convert("RGB"), (w, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 10), f"GT ({len(gt_items)})", fill=(255, 255, 255), font=font)
    draw.text((w + 10, 10), f"Pred ({len(preds)})", fill=(255, 255, 255), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def init_stats(nc):
    return {
        "tp": np.zeros(nc, dtype=np.int64),
        "fp": np.zeros(nc, dtype=np.int64),
        "fn": np.zeros(nc, dtype=np.int64),
        "loc_tp": np.zeros(nc, dtype=np.int64),
        "loc_fn": np.zeros(nc, dtype=np.int64),
        "matched_iou": defaultdict(list),
        "timing": {"num_images": 0, "forward_sec": 0.0, "total_sec": 0.0},
    }


def evaluate_image(preds, gt_items, nc, stats):
    if not preds:
        for g in gt_items:
            stats["fn"][int(g["label"])] += 1
            stats["loc_fn"][int(g["label"])] += 1
        return
    if not gt_items:
        for p in preds:
            stats["fp"][int(p["pred_class"])] += 1
        return

    iou_mat = pairwise_mask_iou([p["mask"] for p in preds], [g["mask"] for g in gt_items])
    rows, cols = linear_sum_assignment(1.0 - iou_mat)
    matched_pred, matched_gt = set(), set()

    for r, c in zip(rows, cols):
        if float(iou_mat[r, c]) < 0.5:
            continue
        matched_pred.add(int(r))
        matched_gt.add(int(c))
        pc = int(preds[r]["pred_class"])
        gc = int(gt_items[c]["label"])
        stats["loc_tp"][gc] += 1
        if pc == gc:
            stats["tp"][gc] += 1
            stats["matched_iou"][gc].append(float(iou_mat[r, c]))
        else:
            stats["fp"][pc] += 1
            stats["fn"][gc] += 1

    for i in range(len(preds)):
        if i not in matched_pred:
            stats["fp"][int(preds[i]["pred_class"])] += 1
    for j, g in enumerate(gt_items):
        gc = int(g["label"])
        if j not in matched_gt:
            stats["fn"][gc] += 1
            stats["loc_fn"][gc] += 1


def predictions_to_coco(preds, img_id, cat_inv):
    out = []
    for p in preds:
        pc = int(p["pred_class"])
        if pc not in cat_inv:
            continue
        bbox = [float(v) for v in p["box"]]
        xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        m2d = np.squeeze(p["mask"].astype(np.uint8))
        rle = maskutil.encode(np.asfortranarray(m2d))
        rle["counts"] = rle["counts"].decode("utf-8")
        out.append({
            "image_id": int(img_id), "category_id": int(cat_inv[pc]),
            "bbox": xywh, "score": float(p["score"]),
            "segmentation": rle, "area": float(m2d.sum()), "iscrowd": 0,
        })
    return out


def fmt_opt(x, width=10, ndigits=4):
    if x is None:
        return " " * width
    return f"{float(x):>{width}.{ndigits}f}"


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(args, cfg, model, processor, device):
    ann_file = Path(cfg["dataset"]["root"]) / args.split / "_annotations.coco.json"
    img_dir = Path(cfg["dataset"]["root"]) / args.split

    if not ann_file.exists():
        raise FileNotFoundError(f"Not found: {ann_file}")

    with open(ann_file) as f:
        coco = json.load(f)

    class_names = cfg["dataset"]["class_names"]
    nc = len(class_names)

    norm_to_idx = {normalize_name(n): i for i, n in enumerate(class_names)}
    cat_map = {}
    for c in coco["categories"]:
        key = normalize_name(c["name"])
        if key in norm_to_idx:
            cat_map[int(c["id"])] = int(norm_to_idx[key])
    cat_inv = {v: k for k, v in cat_map.items()}
    coco_cat_ids = list(cat_map.keys())

    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann["category_id"]) in cat_map:
            img_to_anns[int(ann["image_id"])].append(ann)

    img_id_to_info = {int(img["id"]): img for img in coco["images"]}
    image_ids = list(img_id_to_info.keys())

    stats = init_stats(nc)
    all_coco_preds = []

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qual_dir = out_dir / f"{args.exp_tag}_qualitative_{args.split}"
    if args.save_qualitative:
        qual_dir.mkdir(parents=True, exist_ok=True)

    nms_thr = float(args.nms_iou_threshold)

    print(
        f"\nEvaluation | split={args.split} | images={len(image_ids)} | "
        f"score_thr={args.score_threshold} | nms_thr={nms_thr}"
    )

    qual_saved = 0

    for img_id in tqdm(image_ids, desc="Evaluating"):
        info = img_id_to_info[img_id]
        h, w = int(info["height"]), int(info["width"])
        path = img_dir / info["file_name"]
        if not path.exists():
            continue

        pil = Image.open(path).convert("RGB")
        t_start = time.perf_counter()
        fwd_time = 0.0

        # ── One text prompt per class on the full image ──
        all_preds = []
        for class_idx, class_name in enumerate(class_names):
            t0 = time.perf_counter()
            preds = collect_prompt_predictions(
                model=model, processor=processor, device=device,
                pil=pil, class_name=class_name, class_idx=class_idx,
                h=h, w=w,
                score_thr=float(args.score_threshold),
                mask_thr=float(args.mask_threshold),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            fwd_time += time.perf_counter() - t0
            all_preds.extend(preds)

        # ── Class-agnostic NMS ──
        all_preds = class_agnostic_mask_nms(all_preds, iou_thr=nms_thr)

        # ── COCO predictions ──
        if all_preds:
            all_coco_preds.extend(predictions_to_coco(all_preds, img_id, cat_inv))

        # ── GT items ──
        gt_items = []
        for ann in img_to_anns.get(img_id, []):
            mask = ann_to_mask(ann, h, w)
            if mask is None or int(mask.sum()) <= 0:
                continue
            x, y, bw, bh = ann["bbox"]
            gt_items.append({
                "box": np.array([x, y, x + bw, y + bh], dtype=np.float32),
                "mask": mask.astype(np.uint8),
                "label": int(cat_map[int(ann["category_id"])]),
            })

        evaluate_image(all_preds, gt_items, nc, stats)

        t_end = time.perf_counter()
        stats["timing"]["num_images"] += 1
        stats["timing"]["forward_sec"] += fwd_time
        stats["timing"]["total_sec"] += (t_end - t_start)

        # ── Qualitative ──
        if args.save_qualitative:
            if args.qualitative_max < 0 or qual_saved < args.qualitative_max:
                save_qualitative_image(
                    qual_dir / f"{Path(info['file_name']).stem}.jpg",
                    pil, all_preds, gt_items, class_names,
                )
                qual_saved += 1

    # ══════════════════════════════════════════════════════════════════
    # Print results
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "-" * 140)
    print(
        f"{'Class':22s} {'mIoU@TP':>10s} {'mIoU@all':>10s} "
        f"{'Precision':>10s} {'Recall':>10s} {'LocRec50':>10s} "
        f"{'ClsF1':>10s} {'TP':>6s} {'FP':>6s} {'FN':>6s}"
    )
    print("-" * 140)

    results = {}
    agg = {k: [] for k in ("miou", "miou_all", "prec", "rec", "locrec", "f1")}

    for c, name in enumerate(class_names):
        tp = int(stats["tp"][c])
        fp = int(stats["fp"][c])
        fn = int(stats["fn"][c])
        loc_tp = int(stats["loc_tp"][c])
        loc_fn = int(stats["loc_fn"][c])

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
        locrec = loc_tp / (loc_tp + loc_fn + 1e-8)

        ious = stats["matched_iou"].get(c, [])
        miou = float(np.mean(ious)) if ious else None
        # mIoU@all: every ground-truth object of this class without a TP counts as IoU = 0
        n_gt = tp + fn
        miou_all = float(np.sum(ious)) / n_gt if n_gt > 0 else None

        if miou: agg["miou"].append(miou)
        if miou_all is not None: agg["miou_all"].append(miou_all)
        for k, v in [("prec", prec), ("rec", rec), ("locrec", locrec), ("f1", f1)]:
            agg[k].append(v)

        print(
            f"{name:22s} {fmt_opt(miou)} {fmt_opt(miou_all)} "
            f"{prec:10.4f} {rec:10.4f} {locrec:10.4f} {f1:10.4f} "
            f"{tp:6d} {fp:6d} {fn:6d}"
        )
        results[name] = {
            "mIoU@TP": miou, "mIoU@all": miou_all,
            "Precision": float(prec), "Recall": float(rec),
            "LocRec50": float(locrec), "ClsF1": float(f1),
            "TP": tp, "FP": fp, "FN": fn,
        }

    print("-" * 140)
    ni = max(stats["timing"]["num_images"], 1)
    m = {k: float(np.mean(v)) if v else None for k, v in agg.items()}
    print(
        f"{'MEAN':22s} {fmt_opt(m['miou'])} {fmt_opt(m['miou_all'])} "
        f"{m['prec']:.4f} {m['rec']:.4f} {m['locrec']:.4f} {m['f1']:.4f}"
    )

    # ── COCO metrics ──
    coco_stats = None
    if all_coco_preds:
        pred_file = out_dir / f"{args.exp_tag}_predictions_{args.split}.json"
        with open(pred_file, "w") as f:
            json.dump(all_coco_preds, f)

        coco_gt = COCO(str(ann_file))
        coco_dt = coco_gt.loadRes(str(pred_file))
        ce = COCOeval(coco_gt, coco_dt, "segm")
        ce.params.catIds = coco_cat_ids
        ce.params.imgIds = image_ids
        ce.evaluate()
        ce.accumulate()
        ce.summarize()
        st = ce.stats
        coco_stats = {
            "AP": float(st[0]), "AP50": float(st[1]), "AP75": float(st[2]),
            "APsmall": float(st[3]), "APmedium": float(st[4]), "APlarge": float(st[5]),
            "AR100": float(st[8]),
        }
        print(f"\nCOCO segm:")
        print(f"  AP@[.50:.95] = {coco_stats['AP']:.4f}")
        print(f"  AP50 = {coco_stats['AP50']:.4f}")
        print(f"  AP75 = {coco_stats['AP75']:.4f}")
        print(f"  AR@100 = {coco_stats['AR100']:.4f}")

    print(f"\nTiming: fwd={stats['timing']['forward_sec']/ni:.3f}s/img | "
          f"total={stats['timing']['total_sec']/ni:.3f}s/img")
    if args.save_qualitative:
        print(f"Qualitative: {qual_saved} images → {qual_dir}")

    # ── Save ──
    final = {
        "args": vars(args),
        "eval_mode": "one_prompt_per_class_full_image",
        "per_class": results,
        "mean": m,
        "coco": coco_stats,
    }
    out_path = out_dir / f"{args.exp_tag}_eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate SAM 3 with LoRA (Text-Full-D)"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--exp-tag", default="text_full_d")
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--score-threshold", type=float, default=0.15)
    ap.add_argument("--mask-threshold", type=float, default=0.50)
    ap.add_argument("--nms-iou-threshold", type=float, default=0.50)
    ap.add_argument("--save-qualitative", action="store_true")
    ap.add_argument("--qualitative-max", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["hardware"].get("seed", 42)))
    device = get_device()

    model = build_model(cfg).to(device)
    processor = build_processor(cfg)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Model loaded | missing={len(missing)} | unexpected={len(unexpected)}")

    model.eval()
    evaluate(args, cfg, model, processor, device)


if __name__ == "__main__":
    main()