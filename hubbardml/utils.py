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


def calculate_u_energy(hubbard_u: float, occs: np.ndarray) -> float:
    return hubbard_u * (np.sum(np.trace(occs)) - np.einsum("ijs,jis", occs, occs))


def calculate_v_energy(hubbard_v: float, occupations: np.ndarray) -> float:
    return hubbard_v * np.einsum("ijs,jis", occupations, occupations)


def calculate_hubbard_energy(hubbard_u: float, hubbard_v: float, occupations: np.ndarray) -> float:
    return calculate_u_energy(hubbard_u, occupations) + calculate_v_energy(hubbard_v, occupations)
