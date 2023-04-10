import abc
import collections
import functools
import math
import pathlib
import uuid
from typing import Dict, Tuple, Iterable, Union, List

from e3nn import o3
import e3psi
import numpy as np
import pandas as pd
import torch

from . import datasets
from . import keys
from . import plots
from . import sites
from . import similarities

__all__ = "VGraph", "UGraph", "UVGraph", "UGraph", "UModel", "VModel"

SPECIES = "species"
U = "U"
V = "V"
UV = "UV"


def _prepare_dataset(df: pd.DataFrame):
    # Put in the self-consistent paths
    df[keys.SC_PATHS] = df.apply(lambda row: str(pathlib.Path(row[keys.DIR]).parent), axis=1)

    return df


# Convenience function for finding the correct column based on site and occupation index
def key(prop: str, atom_idx: int, occs_idx: int = None):
    if occs_idx is None:
        return f"atom_{atom_idx}_{prop}"

    return f"atom_{atom_idx}_{prop}_{occs_idx}"


class TensorSum(e3psi.Attr):
    """
    Sum the input tensors.  Useful if you want to create a permutationally invariant representation.
    """

    def __init__(self, attr: e3psi.Attr):
        self._attr = attr
        super().__init__(self._attr.irreps)

    def create_tensor(self, value: List, dtype=None, device=None) -> torch.Tensor:
        create = functools.partial(self._attr.create_tensor, dtype=dtype, device=device)
        return sum(map(create, value))


class TensorElementwiseProduct(e3psi.Attr):
    """
    Elementwise tensor product. Useful if you want to create a permutationally invariant representation.
    """

    def __init__(self, attr: e3psi.Attr, filter_ir_out=None):
        self._attr = attr
        self._tp = o3.ElementwiseTensorProduct(
            attr.irreps,
            attr.irreps,
            filter_ir_out=filter_ir_out,
            irrep_normalization="norm",
        )
        super().__init__(self._tp.irreps_out)

    def create_tensor(self, value: List, dtype=None, device=None) -> torch.Tensor:
        create = functools.partial(self._attr.create_tensor, dtype=dtype, device=device)
        occs_1, occs_2 = tuple(map(create, value))
        self._tp.to(device=device, dtype=dtype)
        return self._tp(occs_1, occs_2)


class Site(sites.Site):
    """A site that carried information about the species and permutationally invariant occupations matrix tensors"""

    def __init__(self, species: Iterable[str], occ_irreps: Union[str, o3.Irrep]) -> None:
        super().__init__()
        self.specie = e3psi.SpecieOneHot(species)
        self._occs = e3psi.OccuMtx(occ_irreps)  # The occupations matrix representation
        self.occs_sum = TensorSum(self._occs)
        self.occs_prod = TensorElementwiseProduct(self._occs)  # , filter_ir_out=["0e", "2e"])


class ModelGraph(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_inputs(self, raw_data, dtype=None, device=None) -> Dict:
        """Create the inputs for a model"""


class UGraph(e3psi.graphs.OneSite, ModelGraph):
    """A graph that contains only one (d-element) site"""

    TYPE_ID = uuid.UUID("4d7951c8-5fc7-4c9d-883e-ef09d27f478c")

    DEFAULT_GROUP_BY = (keys.SC_PATHS, keys.ATOM_1_ELEMENT)

    def __init__(self, species: Iterable[str]) -> None:
        self.species = tuple(species)
        dsite = Site(species, "2e")  # D site
        super().__init__(dsite)

    def create_inputs(self, raw_data, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        occupations = datasets.get_occupation_matrices(raw_data, 1)
        site_tensor = self.site.create_tensor(
            dict(
                specie=raw_data[keys.ATOM_1_ELEMENT],
                # Pass the same up/down occupation matrices to both the sum and product
                occs_sum=occupations,
                occs_prod=occupations,
            ),
            dtype=dtype,
            device=device,
        )
        return dict(site=site_tensor)

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

        df = _prepare_dataset(df)

        return datasets.filter_dataset(
            df,
            param_type=keys.PARAM_U,
            remove_vwd=True,
            remove_zero_out=True,
            remove_in_eq_out=False,
        )

    def get_similarity_frame(self, data: pd.DataFrame, group_by=DEFAULT_GROUP_BY):
        data = data.copy()

        data[keys.SC_PATHS] = data.apply(
            lambda row: str(pathlib.Path(row[keys.DIR]).parent), axis=1
        )

        # Create the power spectrum distance comparator
        input_irreps = {"site": e3psi.irreps(self.site)}
        dist_ps = e3psi.distances.PowerSpectrumDistance(input_irreps)
        power_spectra = data.apply(
            lambda row: dist_ps.power_spectrum(self.create_inputs(row)),
            axis=1,
            result_type="reduce",
        )

        similarity_data = []
        similarity_cols = (
            similarities.SimilarityKeys.INDEX_PAIR,
            similarities.SimilarityKeys.INPUT_DIST,
            similarities.SimilarityKeys.DIST_TRACE,
            similarities.SimilarityKeys.DELTA_PARAM,
        )

        for _name, indexes in data.groupby(list(group_by)).groups.items():
            for i, idx_i in enumerate(indexes):
                row_i = data.loc[idx_i]
                ps_i = power_spectra.loc[idx_i]

                for j_, idx_j in enumerate(indexes[i + 1 :]):
                    row_j = data.loc[idx_j]
                    ps_j = power_spectra.loc[idx_j]

                    dist_ps_ij = dist_ps.get_distance_from_ps(ps_i, ps_j)

                    dist_trace = abs(
                        (row_i[keys.ATOM_1_OCCS_1].trace() + row_i[keys.ATOM_1_OCCS_2].trace())
                        - (row_j[keys.ATOM_1_OCCS_1].trace() + row_j[keys.ATOM_1_OCCS_2].trace())
                    )

                    delta_param = abs(row_i[keys.PARAM_OUT] - row_j[keys.PARAM_OUT])

                    similarity_data.append([{idx_i, idx_j}, dist_ps_ij, dist_trace, delta_param])

        return pd.DataFrame(similarity_data, columns=similarity_cols)


class UModel(e3psi.models.OnsiteModel):
    """Hubbard +U model

    Consist of a single node that carries information about the atomic specie and occupation matrix.
    """

    TYPE_ID = uuid.UUID("947373b7-6cfc-42f6-bbcb-279f88a02db2")

    def __init__(
        self,
        graph: UGraph,
        feature_irreps="23x0e + 4x2e + 1x3e + 4x4e + 1x5e + 2x6e + 1x8e",
        hidden_layers=2,
        **kwargs,
    ) -> None:
        if feature_irreps is None:
            feature_irreps = self._compress_non_scalars(
                o3.ReducedTensorProducts("ij=ji", i=e3psi.irreps(graph.site)).irreps_out, factor=1.0
            )

        super().__init__(
            graph,
            feature_irreps=feature_irreps,
            irreps_out="0e",
            hidden_layers=hidden_layers,
            **kwargs,
        )

        self._species = graph.species

    @property
    def species(self) -> Tuple[str]:
        """Get the supported species"""
        return self._species

    def _compress_non_scalars(self, irreps: o3.Irreps, factor=3.0) -> o3.Irreps:
        return o3.Irreps(
            o3._irreps._MulIr(int(math.ceil(float(mul_ir.mul) / factor)), mul_ir.ir)
            if mul_ir.ir.l != 0
            else mul_ir
            for mul_ir in irreps
        )


class VGraph(e3psi.TwoSite, ModelGraph):
    """A two-site model for Hubbard V interactions"""

    uuid.UUID("be7f3ec5-7412-4a98-a71a-2ccf16e27dd1")

    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"
    DEFAULT_GROUP_BY = (keys.SC_PATHS, P_ELEMENT, D_ELEMENT)

    def __init__(self, species: Iterable[str]):
        # Create the graph by supplying the sites
        self._species = tuple(species)
        psite = Site(species, "1o")
        dsite = Site(species, "2e")
        super().__init__(psite, dsite, sites.VEdge())

    @property
    def species(self) -> Tuple[str]:
        return self._species

    def get_pd_idx(self, row):
        if row[keys.ATOM_1_OCCS_1].shape[0] == 3:
            # Assume that atom 1 is the p-block atom and 2 is the d-block atom
            return 1, 2
        else:
            # Assume that atom 2 is the p-block atom and 1 is the d-block atom
            return 2, 1

    def create_inputs(self, row, dtype=None, device=None) -> Dict:
        pidx, didx = self.get_pd_idx(row)
        kwargs = dict(dtype=dtype, device=device)

        # Do some sanity checks
        _check_shape(row, key("occs_inv", pidx, 1), (3, 3))
        _check_shape(row, key("occs_inv", pidx, 2), (3, 3))

        _check_shape(row, key("occs_inv", didx, 1), (5, 5))
        _check_shape(row, key("occs_inv", didx, 2), (5, 5))

        p_occupations = datasets.get_occupation_matrices(row, pidx)
        site1_tensor = e3psi.create_tensor(
            self.site1,
            dict(
                specie=row[key("element", pidx)],
                occs_sum=p_occupations,
                occs_prod=p_occupations,
            ),
            **kwargs,
        )

        d_occupations = datasets.get_occupation_matrices(row, didx)
        site2_tensor = e3psi.create_tensor(
            self.site2,
            dict(
                specie=row[key("element", didx)],
                occs_sum=d_occupations,
                occs_prod=d_occupations,
            ),
            **kwargs,
        )

        edge_tensor = e3psi.create_tensor(
            self.edge, dict(one=1, v=row[keys.PARAM_IN], dist=row[keys.DIST_IN]), **kwargs
        )

        return dict(
            site1=site1_tensor,
            site2=site2_tensor,
            edge=edge_tensor,
        )

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
        df = _prepare_dataset(df)

        return df

    def get_similarity_frame(self, data: pd.DataFrame, group_by=DEFAULT_GROUP_BY):
        data = data.copy()

        data[keys.SC_PATHS] = data.apply(
            lambda row: str(pathlib.Path(row[keys.DIR]).parent), axis=1
        )

        # Create the power spectrum distance comparator
        input_irreps = {
            name: e3psi.irreps(getattr(self, name)) for name in ("site1", "site2", "edge")
        }
        dist_ps = e3psi.distances.PowerSpectrumDistance(input_irreps)
        power_spectra = data.apply(
            lambda row: dist_ps.power_spectrum(self.create_inputs(row)),
            axis=1,
            result_type="reduce",
        )

        similarity_data = []
        similarity_cols = (
            similarities.SimilarityKeys.INDEX_PAIR,
            similarities.SimilarityKeys.INPUT_DIST,
            similarities.SimilarityKeys.DIST_TRACE,
            similarities.SimilarityKeys.DELTA_PARAM,
        )

        for _name, indexes in data.groupby(list(group_by)).groups.items():
            for i, idx_i in enumerate(indexes):
                row_i = data.loc[idx_i]
                ps_i = power_spectra.loc[idx_i]

                for j_, idx_j in enumerate(indexes[i + 1 :]):
                    row_j = data.loc[idx_j]
                    ps_j = power_spectra.loc[idx_j]

                    dist_ps_ij = dist_ps.get_distance_from_ps(ps_i, ps_j)

                    # dist_trace = abs((row_i[self.P_ELEMENT].trace() + row_i[keys.ATOM_1_OCCS_2].trace()) - (
                    #         row_j[keys.ATOM_1_OCCS_1].trace() + row_j[keys.ATOM_1_OCCS_2].trace()))

                    dist_trace = 0.0  # TODO: Implement this

                    delta_param = abs(row_i[keys.PARAM_OUT] - row_j[keys.PARAM_OUT])

                    similarity_data.append([{idx_i, idx_j}, dist_ps_ij, dist_trace, delta_param])

        return pd.DataFrame(similarity_data, columns=similarity_cols)


class VModel(e3psi.IntersiteModel):
    """Hubbard +V model"""

    TYPE_ID = uuid.UUID("4a647f9f-0f90-474a-928e-19691b576597")
    # Columns to store the current p-block and d-block elements
    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"

    def __init__(
        self,
        graph: VGraph,
        node_features="10x0e+4x1e+6x2e",
        **kwargs,
    ):
        super().__init__(
            graph,
            node_features=node_features,
            irreps_out="0e",
            **kwargs,
        )
        self._species = tuple(graph.species)

    @property
    def species(self) -> Tuple:
        """Get the supported species"""
        return self._species


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
        node_features="106x0e+4x1e+6x2e+8x3e+4x4e+2x5e+2x6e",
    ):
        super().__init__(
            UVGraph(species),
            node_features=node_features,
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
        inp = graph.create_inputs(row, dtype=dtype, device=device)
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
    U: UModel,
    V: VModel,
    UV: UVModel,
}

GRAPHS = {
    U: UGraph,
    V: VGraph,
    UV: UVGraph,
}

HISTORIAN_TYPES = VModel, UModel
