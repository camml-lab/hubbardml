import random

import numpy as np
import torch


def rmse(y1, y2) -> float:
    return np.sqrt(((y1 - y2) ** 2).mean())


def random_seed(seed: int = 0xDEADBEEF):
    random.seed(seed)
    torch.manual_seed(seed)


def _count():
    """Count up from 0 to infinity (and beyond)"""
    i = 0
    while True:
        yield i
        i += 1
