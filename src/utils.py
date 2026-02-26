"""
Utilitários gerais: seed, device detection, timing.
"""

import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

from src.config import SEED


def set_seed(seed: int = SEED) -> None:
    """Fixa a seed para reprodutibilidade em todas as bibliotecas."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


def get_device() -> torch.device:
    """Detecta o melhor device disponível (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def timer(description: str = "Operação"):
    """Context manager para medir tempo de execução."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    if minutes > 0:
        print(f"⏱ {description}: {minutes}min {seconds:.1f}s")
    else:
        print(f"⏱ {description}: {seconds:.1f}s")
