"""
utils.py

Helper functions: random seed, logging, config loading and saving, device
selection, and early stopping.
"""

import sys
import yaml
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark    = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False


def setup_logging(output_dir: Path, log_file: str = "training.log") -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("SAM3_LoRA")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / log_file, mode="a")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Logging initialized. Log file: {output_dir / log_file}")
    return logger


def load_config(config_path: str) -> Dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from: {config_path}")
    return cfg


def save_config(config: Dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        props    = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        print("\n" + "=" * 60)
        print("GPU INFORMATION")
        print("=" * 60)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {total_gb:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        try:
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        except Exception:
            pass
        print("=" * 60 + "\n")
        return device
    print("\nWARNING: CUDA not available. Running on CPU.\n")
    return torch.device("cpu")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.,
                 mode: str = "max", verbose: bool = True):
        self.patience   = int(patience)
        self.min_delta  = float(min_delta)
        self.mode       = str(mode)
        self.verbose    = bool(verbose)
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        if mode == "max":
            self._is_better = lambda n, b: n > b + self.min_delta
        elif mode == "min":
            self._is_better = lambda n, b: n < b - self.min_delta
        else:
            raise ValueError("mode must be 'max' or 'min'")

    def __call__(self, metric: float) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False
        if self._is_better(metric, self.best_score):
            self.best_score = metric
            self.counter    = 0
            return False
        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print("Early stopping triggered!")
            return True
        return False

    def reset(self):
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
