import math
import pathlib
import subprocess
import types

from e3nn import o3
import numpy as np
import pytest
import torch

from hubbardml import qe


@pytest.fixture(scope="session")
def qe_module() -> types.ModuleType:
    try:
        import ylmr2
    except ImportError:
        cwd = pathlib.Path(__file__).parent.absolute()
        in_path = cwd / "ylmr2.f90"

        command = list(f"f2py -c {in_path} -m ylmr2 --build-dir {cwd}".split(" "))
        subprocess.run(command)
        from tests import ylmr2

    return ylmr2


def s1_grid():
    theta = torch.linspace(0, 2 * math.pi, 40)
    return torch.vstack((np.cos(theta), np.sin(theta))).T


def s2_grid(dtype=None):
    betas = torch.linspace(0, math.pi, 40, dtype=dtype)
    alphas = torch.linspace(0, 2 * math.pi, 80, dtype=dtype)
    beta, alpha = torch.meshgrid(betas, alphas, indexing="ij")
    return o3.angles_to_xyz(alpha, beta)


def test_qe_cob(qe_module: types.ModuleType):
    pts = s2_grid(torch.float64).reshape(-1, 3)
    lmax = 10
    ylm_outs = qe_module.ylmr2((lmax + 1) ** 2, np.array(pts, order="F").T, np.ones(len(pts)))

    for ell in range(lmax + 1):
        l2 = ell**2
        next_l2 = (ell + 1) ** 2
        e3_ylm = o3.spherical_harmonics(ell, pts, True)  # From e3nn
        e3_ylm_standard = qe.e3_to_standard(ell, e3_ylm)

        qe_ylm = torch.tensor(ylm_outs[:, l2:next_l2], dtype=e3_ylm.dtype)  # From QE
        qe_ylm_standard = qe.qe_to_standard(ell, qe_ylm)

        assert torch.allclose(qe_ylm_standard, e3_ylm_standard, atol=1e-5)


def test_qe_to_e3(qe_module: types.ModuleType):
    pts = s2_grid(torch.float64).reshape(-1, 3)
    lmax = 10
    ylm_outs = qe_module.ylmr2((lmax + 1) ** 2, np.array(pts, order="F").T, np.ones(len(pts)))

    for ell in range(lmax + 1):
        l2 = ell**2
        next_l2 = (ell + 1) ** 2
        e3_ylm = o3.spherical_harmonics(ell, pts, True)  # From e3nn

        qe_ylm = torch.tensor(ylm_outs[:, l2:next_l2], dtype=e3_ylm.dtype)  # From QE
        qe_ylm_e3 = qe.qe_to_e3(ell, qe_ylm)

        assert torch.allclose(qe_ylm_e3, e3_ylm, atol=1e-5)
