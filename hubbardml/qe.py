"""
Module to store utility functions related to Quantum ESPRESSO

The definition and order of QE spherical harmonics needs to be taken into account.  A description of their convention
can be found here:

    https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html#idm100


We use 'standard' real spherical harmonics as defined here:

    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

as the reference to which everything is transformed, and then we can transform from this back to e3nn or QE convention.
"""

import torch
from e3nn import o3
import e3psi


# E3NN change of basis functions

# Define the change of basis zxy -> xyz
# e3nn follows x, y, z convention
change_of_basis = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def e3_to_standard_cob(l: int, dtype=None):  # noqa: E741
    return o3.Irrep(l, (-1) ** l).D_from_matrix(change_of_basis.to(dtype=dtype))


def e3_to_standard(l: int, ylm):  # noqa: E741
    cob = e3_to_standard_cob(l, dtype=ylm.dtype)
    return ylm @ cob


# QE conversion functions


def qe_to_standard_m(l: int):  # noqa: E741
    yield 0  # m = 0 is always first in QE
    for m in range(1, l + 1):
        # Then QE does the positive followed by negative m
        yield m
        yield -m


def qe_to_standard_cob(
    l: int, remove_condon_shortley=True, dtype=None
) -> torch.Tensor:  # noqa: E741
    ms = torch.tensor(list(qe_to_standard_m(l)), dtype=dtype)
    if remove_condon_shortley:
        prefac = (-1) ** (ms)
    else:
        prefac = torch.ones(2 * l + 1)

    ind_order = torch.argsort(ms)
    cob = prefac * torch.nn.functional.one_hot(ind_order).to(dtype=dtype)
    return cob.T


def qe_to_standard(
    l: int, ylm: torch.Tensor, remove_condon_shortley=True  # noqa: E741
) -> torch.Tensor:
    cob = qe_to_standard_cob(l, remove_condon_shortley, dtype=ylm.dtype)
    return ylm @ cob


def qe_to_e3_cob(l: int, dtype=None) -> torch.Tensor:  # noqa: E741
    # Here we exploit the fact that these are orthogonal matrices so A^-1 = A^T allows us to quickly
    # get the transformation from 'standard' real spherical harmonics to e3nn's convention
    return qe_to_standard_cob(l, dtype=dtype) @ e3_to_standard_cob(l, dtype=dtype).T


def qe_to_e3(l: int, ylm: torch.Tensor) -> torch.Tensor:  # noqa: E741
    return ylm @ qe_to_e3_cob(l, dtype=ylm.dtype)


class QeOccuMtx(e3psi.OccuMtx):
    def __init__(self, orbital_irrep) -> None:
        super().__init__(orbital_irrep)
        self._occ_irrep = o3.Irrep(orbital_irrep)

    def create_tensor(self, occ_mtx, dtype=None, device=None) -> torch.Tensor:
        dtype = dtype or torch.get_default_dtype()

        # First perform the change of basis from QE to e3nn
        cob = qe_to_e3_cob(self._occ_irrep.l, dtype=dtype).to(
            device=device
        )  # Get the change of basis matrix
        occ_mtx = cob.T @ torch.tensor(occ_mtx, device=device, dtype=dtype) @ cob
        # Now, create the spherical tensor
        return super().create_tensor(occ_mtx, dtype, device)
