import logging
import random
from typing import Iterator, Tuple

import numpy as np
import numpy.random
import torch

_LOGGER = logging.getLogger(__name__)


def rmse(y1, y2):
    return ((y1 - y2) ** 2).mean() ** 0.5


def random_seed(seed: int = 0xDEADBEEF):
    _LOGGER.info("Setting random seed to %i", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def _count():
    """Count up from 0 to infinity (and beyond)"""
    i = 0
    while True:
        yield i
        i += 1


def linear_index_pair(total: int) -> Iterator[Tuple[int, int]]:
    """Yield a sequence of index pairs in the order used by scipy.distance.pdist"""
    for i in range(total):
        for j in range(i + 1, total):
            yield i, j


def calculate_u_energy(hubbard_u: float, occs: np.ndarray) -> float:
    return hubbard_u * (np.sum(np.trace(occs)) - np.einsum("ijs,jis", occs, occs))


def calculate_v_energy(hubbard_v: float, occupations: np.ndarray) -> float:
    return hubbard_v * np.einsum("ijs,jis", occupations, occupations)


def calculate_hubbard_energy(hubbard_u: float, hubbard_v: float, occupations: np.ndarray) -> float:
    return calculate_u_energy(hubbard_u, occupations) + calculate_v_energy(hubbard_v, occupations)


def to_mev_string(energy: float) -> str:
    """Return an energy in meV showing units e.g. '234 meV'"""
    return f"{energy * 1000:.0f} meV"
