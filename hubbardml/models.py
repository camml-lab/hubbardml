import collections
import uuid
from typing import List, Dict, Tuple

import e3psi
import pandas as pd
import torch

from . import keys
from . import sites

__all__ = "VGraph", "UGraph"


class VGraph(e3psi.TwoSite):
    """A two-site model for Hubbard V interactions"""

    uuid.UUID("be7f3ec5-7412-4a98-a71a-2ccf16e27dd1")

    def __init__(self, species: List[str]):
        # Create the graph by supplying the sites
        super().__init__(sites.PSite(species), sites.DSite(species), sites.VEdge())

    def create_input(self, row, dtype=None, device=None) -> Dict:
        if row[keys.ATOM_1_OCCS_1].shape[0] == 3:
            # Assume that atom 1 is the p-block atom and 2 is the d-block atom
            pidx = 1
            didx = 2
        else:
            # Assume that atom 2 is the p-block atom and 1 is the d-block atom
            pidx = 2
            didx = 1

        kwargs = dict(dtype=dtype, device=device)

        # Do some sanity checks
        _check_shape(row, f"atom_{pidx}_occs_1", (3, 3))
        _check_shape(row, f"atom_{pidx}_occs_2", (3, 3))

        _check_shape(row, f"atom_{didx}_occs_1", (5, 5))
        _check_shape(row, f"atom_{didx}_occs_2", (5, 5))

        site1_tensor = self.site1.create_tensor(
            dict(
                one=1,
                specie=row[f"atom_{pidx}_element"],
                occs_1=row[f"atom_{pidx}_occs_1"],
                occs_2=row[f"atom_{pidx}_occs_2"],
            ),
            **kwargs,
        )

        site2_tensor = self.site2.create_tensor(
            dict(
                # one=1,
                specie=row[f"atom_{didx}_element"],
                occs_1=row[f"atom_{didx}_occs_1"],
                occs_2=row[f"atom_{didx}_occs_2"],
            ),
            **kwargs,
        )

        edge_tensor = self.edge.create_tensor(dict(one=1, v=row[keys.PARAM_IN], dist=row[keys.DIST_IN]), **kwargs)

        return dict(
            site1=site1_tensor,
            site2=site2_tensor,
            edge=edge_tensor,
        )


class VModel(e3psi.IntersiteModel):
    """Hubbard +V model"""

    TYPE_ID = uuid.UUID("4a647f9f-0f90-474a-928e-19691b576597")

    def __init__(
        self,
        species,
        n1n2_irreps_out="53x0e+2x1e+6x2e+4x3e+5x4e+2x5e+2x6e",
        n1n2e_irreps_out="106x0e+4x1e+6x2e+8x3e+4x4e+2x5e+2x6e",
    ):
        super().__init__(
            VGraph(species),
            n1n2_irreps_out=n1n2_irreps_out,
            n1n2e_irreps_out=n1n2e_irreps_out,
            irreps_out="0e",
        )
        self._species = tuple(species)

    @property
    def species(self) -> Tuple:
        """Get the supported species"""
        return self._species


class UGraph(e3psi.IrrepsObj):
    """A d-block site"""

    TYPE_ID = uuid.UUID("4d7951c8-5fc7-4c9d-883e-ef09d27f478c")

    def __init__(self, species: List[str]) -> None:
        super().__init__()
        self.site = sites.DSite(species)

    def create_input(self, row, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        site_tensor = self.site.create_tensor(
            dict(
                one=1,
                specie=row[keys.ATOM_1_ELEMENT],
                occs_1=row[keys.ATOM_1_OCCS_INV_1],
                occs_2=row[keys.ATOM_1_OCCS_INV_2],
            ),
            dtype=dtype,
            device=device,
        )
        return dict(site=site_tensor)


class UModel(e3psi.OnsiteModel):
    """Hubbard +U model"""

    TYPE_ID = uuid.UUID("4d7951c8-5fc7-4c9d-883e-ef09d27f478c")

    def __init__(
        self,
        species: List[str],
        nn_irreps_out=None,
    ) -> None:
        super().__init__(
            UGraph(species),
            nn_irreps_out=nn_irreps_out,
            irreps_out="0e",
        )

        self._species = tuple(species)

    @property
    def species(self) -> Tuple[str]:
        """Get the supported species"""
        return self._species


def create_model_inputs(graph, frame: pd.DataFrame, dtype=None, device=None) -> Dict[str, torch.Tensor]:
    """Given a graph this method will create a dictionary with stacked tensors for
    each of the inputs in the given dataframe"""
    inputs = collections.defaultdict(list)

    for _, row in frame.iterrows():
        inp = graph.create_input(row, dtype=dtype, device=device)
        # Append to all inputs
        for key, val in inp.items():
            inputs[key].append(val)

    # Stack everything into torch tensors
    for key, val in inputs.items():
        inputs[key] = torch.vstack(val)

    return inputs


def make_predictions(model: e3psi.Model, frame: pd.DataFrame, dtype=None, device=None):
    inputs = create_model_inputs(model.graph, frame, dtype=dtype, device=device)
    return model(inputs).detach().cpu().numpy().reshape(-1)


def _check_shape(row, key, expected: tuple):
    if not row[key].shape == expected:
        raise ValueError(f"Expected {key} to have shape {expected}, got {row[key].shape}")
