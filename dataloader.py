#!/usr/bin/env python3
"""
dataloader.py

Full-image COCO dataset for training and evaluating SAM 3 with LoRA (Text-Full-D).
Each sample is a whole image with all its annotated instances (polygon masks).

Expected folder layout (set dataset.root in the config):
    <dataset.root>/
        train/   images + _annotations.coco.json
        valid/   images + _annotations.coco.json
        test/    images + _annotations.coco.json

Design:
  - Masks are decoded once when the dataset is created and kept in memory.
  - Augmentation is applied in __getitem__, for the training split only.
  - Flips are applied to both the image and the masks.
  - Color jitter, blur, and grayscale are applied to the image only.
  - Boxes are recomputed from the (augmented) masks.
"""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
except ImportError as e:
    raise ImportError("pycocotools is required.") from e


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def resolve_annotation_file(split_dir: Path) -> str:
    for name in ("_annotations.coco.json", "annotations.coco.json"):
        p = split_dir / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No COCO annotation file in {split_dir}")


def decode_segmentation(ann: Dict, h: int, w: int) -> Optional[np.ndarray]:
    seg = ann.get("segmentation", None)
    if seg is None:
        return None
    if isinstance(seg, dict):
        counts = seg.get("counts", None)
        if counts is None:
            return None
        rle = mask_utils.frPyObjects(seg, h, w) if isinstance(counts, list) else seg
        return mask_utils.decode(rle).astype(np.uint8)
    if isinstance(seg, list) and len(seg) > 0:
        rles = mask_utils.frPyObjects(seg, h, w)
        return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)
    return None


def masks_to_boxes(masks_np: List[np.ndarray]) -> List[List[float]]:
    """Compute tight xyxy boxes from binary masks."""
    boxes = []
    for mask in masks_np:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            boxes.append([0.0, 0.0, 1.0, 1.0])
        else:
            boxes.append([float(xs.min()), float(ys.min()),
                          float(xs.max() + 1), float(ys.max() + 1)])
    return boxes


class FullImageDataset(Dataset):

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        class_names: Optional[List[str]] = None,
        min_mask_area: int = 32,
        augmentation_cfg: Optional[Dict] = None,
    ):
        self.dataset_root = str(dataset_root)
        self.split = str(split)
        self.class_names = list(class_names or [])
        self.num_classes = len(self.class_names)
        self.min_mask_area = int(min_mask_area)

        self.do_augment = (
            self.split == "train"
            and augmentation_cfg is not None
            and bool(augmentation_cfg.get("enabled", False))
        )
        self.aug_cfg = augmentation_cfg if self.do_augment else {}

        self.image_dir = Path(self.dataset_root) / self.split
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Image dir not found: {self.image_dir}")

        ann_file = resolve_annotation_file(self.image_dir)
        print(f"[Dataset] Loading {self.split} from {ann_file}")
        self.coco = COCO(ann_file)

        norm_to_idx = {normalize_name(n): i for i, n in enumerate(self.class_names)}
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.catid_to_class: Dict[int, int] = {}
        for c in cats:
            key = normalize_name(c["name"])
            if key in norm_to_idx:
                self.catid_to_class[int(c["id"])] = int(norm_to_idx[key])

        img_to_anns: Dict[int, List] = defaultdict(list)
        for ann in self.coco.loadAnns(self.coco.getAnnIds()):
            cid = int(ann.get("category_id", -1))
            if cid not in self.catid_to_class:
                continue
            if "segmentation" not in ann:
                continue
            img_to_anns[int(ann["image_id"])].append(ann)

        self.samples: List[Dict] = []
        skipped_img = 0
        skipped_inst = 0

        for img in self.coco.loadImgs(self.coco.getImgIds()):
            img_id = int(img["id"])
            img_path = self.image_dir / img["file_name"]
            if not img_path.exists():
                skipped_img += 1
                continue
            anns = img_to_anns.get(img_id, [])
            if len(anns) == 0:
                continue

            h, w = int(img["height"]), int(img["width"])
            masks_list = []
            labels_list = []

            for ann in anns:
                mask = decode_segmentation(ann, h, w)
                if mask is None or int(mask.sum()) < self.min_mask_area:
                    skipped_inst += 1
                    continue
                masks_list.append(mask)
                labels_list.append(self.catid_to_class[int(ann["category_id"])])

            if len(labels_list) == 0:
                continue

            self.samples.append({
                "image_id": img_id,
                "file_name": img["file_name"],
                "width": w, "height": h,
                "masks": masks_list,
                "labels": labels_list,
            })

        if len(self.samples) == 0:
            raise ValueError(f"No valid images for split={self.split}")

        total_inst = sum(len(s["labels"]) for s in self.samples)
        aug_str = "ON" if self.do_augment else "OFF"
        print(f"[Dataset] {self.split}: {len(self.samples)} images, "
              f"{total_inst} instances, augmentation={aug_str}")
        if skipped_img > 0:
            print(f"[Dataset] Skipped {skipped_img} missing images")

        class_counts = defaultdict(int)
        for s in self.samples:
            for lbl in s["labels"]:
                class_counts[lbl] += 1
        for i, name in enumerate(self.class_names):
            print(f"  {name:<22s} {class_counts.get(i, 0):>6d} instances")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, pil_img, masks_list):
        """Apply training augmentation. Returns the image, the kept masks, and their indices."""
        if not self.do_augment:
            return pil_img, masks_list, list(range(len(masks_list)))

        masks_list = [m.copy() for m in masks_list]

        # Horizontal flip
        if random.random() < float(self.aug_cfg.get("horizontal_flip_p", 0.5)):
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            masks_list = [np.fliplr(m).copy() for m in masks_list]

        # Vertical flip
        if random.random() < float(self.aug_cfg.get("vertical_flip_p", 0.3)):
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            masks_list = [np.flipud(m).copy() for m in masks_list]

        # Color jitter (image only)
        cj = self.aug_cfg.get("color_jitter", {})
        if bool(cj.get("enabled", False)):
            from torchvision.transforms import functional as TF
            b = float(cj.get("brightness", 0.25))
            c = float(cj.get("contrast", 0.25))
            s = float(cj.get("saturation", 0.20))
            h = float(cj.get("hue", 0.02))
            if b > 0:
                pil_img = TF.adjust_brightness(pil_img, max(0.01, 1 + random.uniform(-b, b)))
            if c > 0:
                pil_img = TF.adjust_contrast(pil_img, max(0.01, 1 + random.uniform(-c, c)))
            if s > 0:
                pil_img = TF.adjust_saturation(pil_img, max(0.01, 1 + random.uniform(-s, s)))
            if h > 0:
                pil_img = TF.adjust_hue(pil_img, random.uniform(-h, h))

        # Gaussian blur (image only)
        blur_cfg = self.aug_cfg.get("gaussian_blur", {})
        if bool(blur_cfg.get("enabled", False)):
            if random.random() < float(blur_cfg.get("p", 0.2)):
                ks = blur_cfg.get("kernel_sizes", [3, 5])
                k = random.choice(ks)
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=max(1, (k-1)//2)))

        # Random grayscale (image only)
        if random.random() < float(self.aug_cfg.get("random_grayscale_p", 0.05)):
            pil_img = pil_img.convert("L").convert("RGB")

        # Filter empty masks
        valid_masks = []
        valid_indices = []
        for i, m in enumerate(masks_list):
            if int(m.sum()) >= self.min_mask_area:
                valid_masks.append(m)
                valid_indices.append(i)

        return pil_img, valid_masks, valid_indices

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        image = Image.open(str(self.image_dir / s["file_name"])).convert("RGB")
        h, w = s["height"], s["width"]
        masks_list = s["masks"]
        labels_list = list(s["labels"])

        if self.do_augment:
            image, masks_list, valid_indices = self._augment(image, masks_list)
            labels_list = [labels_list[i] for i in valid_indices]
            if len(masks_list) == 0:
                return self.__getitem__((idx + 1) % len(self))

        # Boxes always recomputed from current masks (handles flips correctly)
        boxes_list = masks_to_boxes(masks_list)

        masks = torch.from_numpy(np.stack(masks_list)).to(torch.uint8)
        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        labels = torch.tensor(labels_list, dtype=torch.long)

        class_present = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in labels_list:
            class_present[lbl] = 1.0

        return {
            "image": image,
            "image_id": int(s["image_id"]),
            "image_file": s["file_name"],
            "orig_size": (h, w),
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "class_present": class_present,
            "num_instances": len(labels_list),
        }


def collate_fn(batch):
    return batch