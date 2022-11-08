import collections
import math
import uuid
from typing import Dict, Tuple, Iterable, Union

from e3nn import o3
import e3psi
import numpy as np
import pandas as pd
import torch

from . import datasets
from . import keys
from . import plots
from . import sites

__all__ = "VGraph", "UGraph", "UVGraph"

SPECIES = "species"
U_MODEL = "U"
V_MODEL = "V"
UV_MODEL = "UV"


# Convenience function for finding the correct column based on site and occupation index
def key(prop: str, atom_idx: int, occs_idx: int = None):
    if occs_idx is None:
        return f"atom_{atom_idx}_{prop}"

    return f"atom_{atom_idx}_{prop}_{occs_idx}"


class UGraph(e3psi.IrrepsObj):
    """A d-block site"""

    TYPE_ID = uuid.UUID("4d7951c8-5fc7-4c9d-883e-ef09d27f478c")

    def __init__(self, species: Iterable[str]) -> None:
        super().__init__()
        self.site = sites.DSite(species)

    def create_input(self, row, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        site_tensor = self.site.create_tensor(
            dict(
                specie=row[keys.ATOM_1_ELEMENT],
                occs_1=row[keys.ATOM_1_OCCS_INV_1],
                occs_2=row[keys.ATOM_1_OCCS_INV_2],
            ),
            dtype=dtype,
            device=device,
        )
        return dict(site=site_tensor)


class UModel(e3psi.models.OnsiteModel):
    """Hubbard +U model

    Consist of a single node that carries information about the atomic specie and occupation matrix.
    """

    TYPE_ID = uuid.UUID("947373b7-6cfc-42f6-bbcb-279f88a02db2")

    def __init__(
        self,
        species: Iterable[str],
        nn_irreps_out="23x0e + 4x2e + 1x3e + 4x4e + 1x5e + 2x6e + 1x8e",
        hidden_layers=2,
        **kwargs,
    ) -> None:
        graph = UGraph(species)
        if nn_irreps_out is None:
            nn_irreps_out = self._compress_non_scalars(
                o3.ReducedTensorProducts("ij=ji", i=graph.irreps).irreps_out, factor=1.0
            )

        super().__init__(
            graph,
            nn_irreps_out=nn_irreps_out,
            irreps_out="0e",
            hidden_layers=hidden_layers,
            **kwargs,
        )

        self._species = tuple(species)

    @property
    def species(self) -> Tuple[str]:
        """Get the supported species"""
        return self._species

    @classmethod
    def prepare_dataset(cls, df: pd.DataFrame) -> pd.DataFrame:
        # Remove non Hubbard active pairs
        for elem in ("O", "S"):
            df = df[~((df[keys.ATOM_1_ELEMENT] == elem) & (df[keys.ATOM_2_ELEMENT] == elem))]

        df[SPECIES] = df.apply(
            lambda row: frozenset([row[keys.ATOM_1_ELEMENT], row[keys.ATOM_2_ELEMENT]]), axis=1
        )
        df[keys.LABEL] = df[keys.ATOM_1_ELEMENT]
        df[keys.COLOUR] = df[keys.ATOM_1_ELEMENT].map(plots.element_colours)

        return datasets.filter_dataset(
            df,
            param_type=keys.PARAM_U,
            remove_vwd=True,
            remove_zero_out=True,
            remove_in_eq_out=False,
        )

    def _compress_non_scalars(self, irreps: o3.Irreps, factor=3.0) -> o3.Irreps:
        return o3.Irreps(
            o3._irreps._MulIr(int(math.ceil(float(mul_ir.mul) / factor)), mul_ir.ir)
            if mul_ir.ir.l != 0
            else mul_ir
            for mul_ir in irreps
        )


class VGraph(e3psi.TwoSite):
    """A two-site model for Hubbard V interactions"""

    uuid.UUID("be7f3ec5-7412-4a98-a71a-2ccf16e27dd1")

    def __init__(self, species: Iterable[str]):
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
        _check_shape(row, key("occs_inv", pidx, 1), (3, 3))
        _check_shape(row, key("occs_inv", pidx, 2), (3, 3))

        _check_shape(row, key("occs_inv", didx, 2), (5, 5))
        _check_shape(row, key("occs_inv", didx, 2), (5, 5))

        site1_tensor = self.site1.create_tensor(
            dict(
                specie=row[key("element", pidx)],
                occs_1=row[key("occs_inv", pidx, 1)],
                occs_2=row[key("occs_inv", pidx, 1)],
            ),
            **kwargs,
        )

        site2_tensor = self.site2.create_tensor(
            dict(
                specie=row[key("element", didx)],
                occs_1=row[key("occs_inv", didx, 1)],
                occs_2=row[key("occs_inv", didx, 2)],
            ),
            **kwargs,
        )

        edge_tensor = self.edge.create_tensor(
            dict(one=1, v=row[keys.PARAM_IN], dist=row[keys.DIST_IN]), **kwargs
        )

        return dict(
            site1=site1_tensor,
            site2=site2_tensor,
            edge=edge_tensor,
        )


class VModel(e3psi.IntersiteModel):
    """Hubbard +V model"""

    TYPE_ID = uuid.UUID("4a647f9f-0f90-474a-928e-19691b576597")
    # Columns to store the current p-block and d-block elements
    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"

    def __init__(
        self,
        species,
        n1n2_irreps_out="53x0e+2x1e+6x2e+4x3e+5x4e+2x5e+2x6e",
        n1n2e_irreps_out="106x0e+4x1e+6x2e+8x3e+4x4e+2x5e+2x6e",
        **kwargs,
    ):
        super().__init__(
            VGraph(species),
            n1n2_irreps_out=n1n2_irreps_out,
            n1n2e_irreps_out=n1n2e_irreps_out,
            irreps_out="0e",
            **kwargs,
        )
        self._species = tuple(species)

    @property
    def species(self) -> Tuple:
        """Get the supported species"""
        return self._species

    @classmethod
    def prepare_dataset(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = datasets.filter_dataset(
            dataframe,
            param_type=keys.PARAM_V,
            remove_vwd=True,
            remove_zero_out=False,
            remove_in_eq_out=False,
        )

        # Exclude all those that have P-P of D-D elements as this model can't deal with those
        df = df[
            ~df.apply(
                lambda row: row[keys.ATOM_1_OCCS_1].shape[0] == row[keys.ATOM_2_OCCS_1].shape[0],
                axis=1,
            )
        ]

        df[SPECIES] = df.apply(
            lambda row: frozenset([row[keys.ATOM_1_ELEMENT], row[keys.ATOM_2_ELEMENT]]), axis=1
        )
        df[cls.P_ELEMENT] = df.apply(
            lambda row: row[keys.ATOM_1_ELEMENT]
            if row[keys.ATOM_1_OCCS_1].shape[0] == 3
            else row[keys.ATOM_2_ELEMENT],
            axis=1,
        )
        df[cls.D_ELEMENT] = df.apply(
            lambda row: row[keys.ATOM_1_ELEMENT]
            if row[keys.ATOM_2_OCCS_1].shape[0] == 3
            else row[keys.ATOM_2_ELEMENT],
            axis=1,
        )
        df[keys.LABEL] = df.apply(lambda row: f"{row[cls.D_ELEMENT]}-{row[cls.P_ELEMENT]}", axis=1)
        df[keys.COLOUR] = df[cls.D_ELEMENT].map(plots.element_colours)

        return df


class UVGraph(e3psi.TwoSite):
    def __init__(self, species: Iterable[str]):
        # Create the graph by supplying the sites
        site = sites.PDSite(species)
        super().__init__(site, site, sites.VEdge())

    def create_input(self, row, dtype=None, device=None) -> Dict:
        kwargs = dict(dtype=dtype, device=device)  # General kwargs passed to create_tensor() calls

        site_tensors = []
        for site_idx in range(1, 3):  # Sites 1 and 2
            occs_shape = row[key("occs_inv", site_idx, 1)].shape
            if occs_shape[0] == 3:
                block_idx = 0
            elif occs_shape[0] == 5:
                block_idx = 1
            else:
                raise ValueError(f"Unexpected occupations matrix shape: {occs_shape}")

            # p, p, d, d
            occs = ((np.ones((3, 3)), np.ones((3, 3))), (np.zeros((5, 5)), np.zeros((5, 5))))
            for occs_idx in range(1, 3):  # Occupations 1 and 2
                occs[block_idx][occs_idx - 1][:] = row[key("occs_inv", site_idx, occs_idx)]

            site_tensor = self.site1.create_tensor(
                dict(
                    specie=row[key("element", site_idx)],
                    p_occs_1=occs[0][0],
                    p_occs_2=occs[0][1],
                    d_occs_1=occs[1][0],
                    d_occs_2=occs[1][1],
                ),
                **kwargs,
            )
            site_tensors.append(site_tensor)

        edge_tensor = self.edge.create_tensor(
            dict(one=1, v=row[keys.PARAM_IN], dist=row[keys.DIST_IN]), **kwargs
        )

        return dict(
            site1=site_tensors[0],
            site2=site_tensors[1],
            edge=edge_tensor,
        )


class UVModel(e3psi.IntersiteModel):
    """Hubbard +U+V model"""

    # Columns to store the current p-block and d-block elements
    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"

    def __init__(
        self,
        species,
        n1n2_irreps_out="53x0e+2x1e+6x2e+4x3e+5x4e+2x5e+2x6e",
        n1n2e_irreps_out="106x0e+4x1e+6x2e+8x3e+4x4e+2x5e+2x6e",
    ):
        super().__init__(
            UVGraph(species),
            n1n2_irreps_out=n1n2_irreps_out,
            n1n2e_irreps_out=n1n2e_irreps_out,
            irreps_out="0e",
        )
        self._species = tuple(species)

    @property
    def species(self) -> Tuple:
        """Get the supported species"""
        return self._species

    @classmethod
    def prepare_dataset(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = datasets.filter_dataset(
            dataframe,
            remove_vwd=True,
            remove_zero_out=False,
            remove_in_eq_out=False,
        )

        df[SPECIES] = df.apply(
            lambda row: frozenset([row[keys.ATOM_1_ELEMENT], row[keys.ATOM_2_ELEMENT]]), axis=1
        )
        df[keys.COLOUR] = df[keys.PARAM_TYPE].map(plots.parameter_colours)
        df[keys.LABEL] = df.apply(
            lambda row: f"{ {row[keys.ATOM_1_ELEMENT], row[keys.ATOM_2_ELEMENT]} }", axis=1
        )

        return df


class Rescaler(e3psi.models.Module):
    def __init__(self, shift=0.0, scale=1.0):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, data):
        # y = mx + c
        return self.scale * data + self.shift

    def inverse(self, data):
        # x = (y - c) / m
        return (data - self.shift) / self.scale

    @classmethod
    def from_data(cls, dataset: Union[np.ndarray, torch.Tensor], method="mean") -> "Rescaler":
        if method == "mean":
            shift = dataset.mean()
            scale = dataset.std()
            return Rescaler(shift, scale)
        elif method == "minmax":
            dmin = dataset.min()
            shift = dmin
            scale = dataset.max() - dmin
            return Rescaler(shift, scale)
        else:
            raise ValueError(f"Unknown method: {method}")


def create_model_inputs(
    graph, frame: pd.DataFrame, dtype=None, device=None
) -> Dict[str, torch.Tensor]:
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


MODELS = {
    U_MODEL: UModel,
    V_MODEL: VModel,
    UV_MODEL: UVModel,
}

HISTORIAN_TYPES = VModel, UModel
