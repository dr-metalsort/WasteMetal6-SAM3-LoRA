"""
model.py

SAM 3 with module-specific LoRA (Text-Full-D).

- All SAM 3 weights are frozen, including the text encoder.
- LoRA is added to attention projections only:
    vision encoder:  q, k, v
    DETR encoder:    q, k, v, o (self- and cross-attention)
    DETR decoder:    q, k, v, o (self-, text cross-, and vision cross-attention)
    mask decoder:    q, k, v, o (prompt cross-attention only)
- Each module group has its own rank (model.lora in the config).
- The model's image size is set to the processor's resize size (dataset.image_size).
"""

import re
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import Sam3Config, Sam3Model, Sam3Processor


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float, group: str):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(linear)}")

        self.linear = linear
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.group = str(group)

        self.lora_A = nn.Parameter(torch.zeros(self.r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, self.r))

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        delta = self.dropout(x) @ self.lora_A.t()
        delta = delta @ self.lora_B.t()
        return base + delta * self.scaling


def _get_parent_module(root: nn.Module, full_name: str):
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _safe_get(outputs, *names):
    for n in names:
        if hasattr(outputs, n):
            return getattr(outputs, n)
    raise AttributeError(f"None of these attributes exist: {names}")


@dataclass
class WrappedInfo:
    name: str
    group: str
    r: int


class SAM3LoRA(nn.Module):
    def __init__(self, backbone: Sam3Model, config: Dict):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.wrapped: List[WrappedInfo] = []

        self._freeze_all()
        self._apply_path_specific_lora()
        self._freeze_text_encoder_explicitly()

    def _freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _freeze_text_encoder_explicitly(self):
        for n, p in self.backbone.named_parameters():
            if n.startswith("text_encoder"):
                p.requires_grad = False

    def _match_group(self, name: str):
        lcfg = self.config["model"]["lora"]

        vision_pat = [
            r"^vision_encoder\.backbone\.layers\.\d+\.attention\.q_proj$",
            r"^vision_encoder\.backbone\.layers\.\d+\.attention\.k_proj$",
            r"^vision_encoder\.backbone\.layers\.\d+\.attention\.v_proj$",
        ]

        detr_enc_pat = [
            r"^detr_encoder\.layers\.\d+\.(self_attn|cross_attn)\.(q_proj|k_proj|v_proj|o_proj)$",
        ]

        detr_dec_pat = [
            r"^detr_decoder\.layers\.\d+\.(self_attn|text_cross_attn|vision_cross_attn)\.(q_proj|k_proj|v_proj|o_proj)$",
        ]

        mask_pat = [
            r"^mask_decoder\.prompt_cross_attn\.(q_proj|k_proj|v_proj|o_proj)$",
        ]

        for p in vision_pat:
            if re.match(p, name):
                return "vision", lcfg["vision"]["r"]

        for p in detr_enc_pat:
            if re.match(p, name):
                return "detr_encoder", lcfg["detr_encoder"]["r"]

        for p in detr_dec_pat:
            if re.match(p, name):
                return "detr_decoder", lcfg["detr_decoder"]["r"]

        for p in mask_pat:
            if re.match(p, name):
                return "mask_decoder", lcfg["mask_decoder"]["r"]

        return None, None

    def _apply_path_specific_lora(self):
        alpha = float(self.config["model"]["lora"]["alpha"])
        dropout = float(self.config["model"]["lora"]["dropout"])

        replace_list = []
        for name, module in self.backbone.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            group, r = self._match_group(name)
            if group is None:
                continue
            replace_list.append((name, module, group, r))

        for name, module, group, r in replace_list:
            parent, child = _get_parent_module(self.backbone, name)
            setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout, group=group))
            self.wrapped.append(WrappedInfo(name=name, group=group, r=r))

        by_group = {}
        for w in self.wrapped:
            by_group[w.group] = by_group.get(w.group, 0) + 1

        print(f"[LoRA] Wrapped {len(self.wrapped)} Linear layers")
        for g, c in by_group.items():
            print(f"  - {g}: {c}")
        for w in self.wrapped[:50]:
            print(f"    {w.name}  [r={w.r}]")
        if len(self.wrapped) > 50:
            print(f"    ... ({len(self.wrapped) - 50} more)")

    def configure_trainable_lora(self, stage: int):
        # Stage 3 (all four module groups) is used for training (train.py).
        allowed = {
            1: {"vision", "detr_encoder"},
            2: {"vision", "detr_encoder", "detr_decoder"},
            3: {"vision", "detr_encoder", "detr_decoder", "mask_decoder"},
        }[int(stage)]

        for m in self.modules():
            if isinstance(m, LoRALinear):
                trainable = m.group in allowed
                m.lora_A.requires_grad = trainable
                m.lora_B.requires_grad = trainable

    def forward_backbone(self, **inputs):
        return self.backbone(**inputs)

    def get_presence_logits(self, outputs):
        return _safe_get(outputs, "presence_logits", "presencelogits")

    def get_pred_logits(self, outputs):
        return _safe_get(outputs, "pred_logits", "predlogits")

    def get_pred_boxes(self, outputs):
        return _safe_get(outputs, "pred_boxes", "predboxes")

    def get_pred_masks(self, outputs):
        return _safe_get(outputs, "pred_masks", "predmasks")


def build_processor(config: Dict):
    hf_id = config["model"]["hf_model_id"]
    image_size = int(config["dataset"]["image_size"])

    processor = Sam3Processor.from_pretrained(
        hf_id,
        size={"height": image_size, "width": image_size},
    )
    return processor


def build_model(config: Dict):
    hf_id = config["model"]["hf_model_id"]
    image_size = int(config["dataset"]["image_size"])

    sam3_config = Sam3Config.from_pretrained(hf_id)
    sam3_config.image_size = image_size

    backbone = Sam3Model.from_pretrained(hf_id, config=sam3_config)
    model = SAM3LoRA(backbone, config)
    model.configure_trainable_lora(stage=1)
    return model

