import abc
import collections
import functools
import operator
import pathlib
from typing import Iterable, Dict, List, Union, Tuple, Mapping
import uuid

from e3nn import o3
import e3psi
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from . import datasets
from . import keys
from . import plots
from . import similarities
from . import sites
from . import utils

__all__ = "ModelGraph", "UGraph", "VGraph", "HubbardDataset"

U = "U"
V = "V"

DEFAULT_OCCS_TOL = 1e-4
DEFAULT_PARAM_TOL = 5e-4  # Input Hubbard parameters less than this are considered to be identical


def diag_mean(mtx):
    return mtx.diagonal().mean()


def key(prop: str, atom_idx: int, occs_idx: int = None):
    """
    Convenience function for finding the correct column based on site and occupation index

    :param prop:
    :param atom_idx:
    :param occs_idx:
    :return:
    """
    if occs_idx is None:
        return f"atom_{atom_idx}_{prop}"

    return f"atom_{atom_idx}_{prop}_{occs_idx}"


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Put in the self-consistent paths
    df[keys.SC_PATHS] = df.apply(lambda row: str(pathlib.Path(row[keys.DIR]).parent), axis=1)

    return df


def _check_shape(row, key, expected: tuple):
    if not row[key].shape == expected:
        raise ValueError(f"Expected {key} to have shape {expected}, got {row[key].shape}")


class TensorSum(e3psi.Attr):
    """
    Sum the input tensors.  Useful if you want to create a permutationally invariant representation.
    """

    def __init__(self, attr: e3psi.Attr):
        self._attr = attr
        super().__init__(self._attr.irreps)

    def create_tensor(self, value: List, dtype=None, device=None) -> torch.Tensor:
        create = functools.partial(self._attr.create_tensor, dtype=dtype, device=device)
        return sum(map(create, value)) / len(value)


class TensorElementwiseProduct(e3psi.Attr):
    """
    Elementwise tensor product. Useful if you want to create a permutationally invariant representation.
    """

    def __init__(self, attr: e3psi.Attr, filter_ir_out=None):
        self._attr = attr
        if filter_ir_out is None:
            base_tp = o3.ElementwiseTensorProduct(
                attr.irreps,
                attr.irreps,
                filter_ir_out=filter_ir_out,
                irrep_normalization="norm",
            )
            filter_ir_out = base_tp.irreps_out

        else:
            filter_ir_out = o3.Irreps(filter_ir_out)

        filter_ir_out = tuple(
            [mul_ir.ir for mul_ir in filter_ir_out.sort().irreps.simplify() if mul_ir.ir.l % 2 == 0]
        )
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
        self._tp.to(device=occs_1.device, dtype=occs_1.dtype)
        return self._tp(occs_1, occs_2)


class Site(sites.Site):
    """A site that carries information about the species and permutationally invariant occupations matrix tensors"""

    def __init__(self, species: Iterable[str], occ_irreps: Union[str, o3.Irrep]) -> None:
        super().__init__()
        self.specie = e3psi.SpecieOneHot(species)
        self._occs = e3psi.OccuMtx(occ_irreps)  # The occupations matrix representation
        self.occs_sum = TensorSum(self._occs)
        self.occs_prod = TensorElementwiseProduct(self._occs)  # , filter_ir_out=["0e", "2e"])

    def create_inputs(self, specie, occs1, occs2, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        occupations = [occs1, occs2]

        # We have to take the absolute value of the occupation matrices here because otherwise we won't
        # be globally invariant to spin flips
        occupations = list(map(lambda occs: np.abs(np.array(occs)), occupations))

        tensor_kwargs = dict(dtype=dtype, device=device)
        return dict(
            specie=e3psi.create_tensor(self.specie, specie, **tensor_kwargs),
            # Pass the same up/down occupation matrices to both the sum and product
            occs_sum=e3psi.create_tensor(self.occs_sum, occupations, **tensor_kwargs),
            occs_prod=e3psi.create_tensor(self.occs_prod, occupations, **tensor_kwargs),
        )


class USite(Site):
    def __init__(self, species: Iterable[str], occ_irreps: Union[str, o3.Irrep]):
        super().__init__(species, occ_irreps)
        self.u_in = o3.Irrep("0e")  # The input Hubbard parameter

    def create_inputs(self, specie, occs1, occs2, u_in, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        inputs = super().create_inputs(specie, occs1, occs2, dtype=dtype, device=device)
        inputs["u_in"] = e3psi.create_tensor(self.u_in, u_in, dtype=dtype, device=device)
        return inputs


class ModelGraph(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_inputs(self, raw_data, dtype=None, device=None) -> Dict:
        """Create the inputs for a model"""

    # @abc.abstractmethod
    def get_similarity_frame(self, dataset: pd.DataFrame, group_by=None) -> pd.DataFrame:
        """Create a data frame containing similarities information"""

    @abc.abstractmethod
    def identify_duplicates(
        self, dataset: pd.DataFrame, group_by, tolerances: dict = None
    ) -> pd.DataFrame:
        """Identify duplicate data in a dataset"""


class UGraph(e3psi.graphs.OneSite, ModelGraph):
    """A graph that contains only one (d-element) site"""

    TYPE_ID = uuid.UUID("4d7951c8-5fc7-4c9d-883e-ef09d27f478c")

    DEFAULT_GROUP_BY = (keys.ATOM_1_ELEMENT,)
    OCCS_TOL = DEFAULT_OCCS_TOL
    PARAM_TOL = DEFAULT_PARAM_TOL

    def __init__(self, species: Iterable[str]) -> None:
        self.species = tuple(species)
        dsite = USite(species, "2e")  # D site
        super().__init__(dsite)

    def create_inputs(self, row: Mapping, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        site_inputs = self.site.create_inputs(
            row[keys.ATOM_1_ELEMENT],
            *datasets.get_occupation_matrices(row, 1),
            row[keys.PARAM_IN],
            dtype=dtype,
            device=device,
        )
        site_tensor = torch.hstack(
            tuple(
                # Use tensorial_attrs() so that the order is guaranteed to be correct
                site_inputs[key]
                for key in e3psi.tensorial_attrs(self.site).keys()
            )
        )
        return dict(site=site_tensor)

    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove non Hubbard active elements
        df = df[df[keys.ATOM_1_ELEMENT].isin(self.species)]
        df = df[df[keys.ATOM_2_ELEMENT].isin(self.species)]

        df[keys.SPECIES] = df.apply(
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

    def get_similarity_frame(self, dataset: pd.DataFrame, group_by=DEFAULT_GROUP_BY):
        power_spectrum_attrs = ("occs_sum", "occs_prod")
        dataset = dataset.copy()

        dataset[keys.SC_PATHS] = dataset.apply(
            lambda row: str(pathlib.Path(row[keys.DIR]).parent), axis=1
        )

        # Create the power spectrum operations that we will use for calculating similarity
        inputs = dataset.apply(
            lambda row: self.site.create_inputs(
                row[keys.ATOM_1_ELEMENT],
                *datasets.get_occupation_matrices(row, 1),
                row[keys.PARAM_IN],
            ),
            axis=1,
            result_type="reduce",
        )
        attrs = e3psi.tensorial_attrs(self.site)
        power_spectra = {}
        for attr_name in power_spectrum_attrs:
            # Get the attribute and create the power spectrum operation for it
            attr = attrs[attr_name]
            dist_ps = e3psi.distances.PowerSpectrumDistance(e3psi.irreps(attr))

            # Calculate The power spectra
            power_spectra[attr_name] = inputs.apply(
                lambda row: dist_ps.power_spectrum(row[attr_name])
            )
        power_spectra = pd.DataFrame(power_spectra)
        similarity_data = collections.defaultdict(list)

        for _name, indexes in dataset.groupby(list(group_by)).groups.items():
            for i, idx_i in enumerate(indexes):
                row_i = dataset.loc[idx_i]
                ps_i = power_spectra.loc[idx_i]

                for j_, idx_j in enumerate(indexes[i + 1 :]):
                    row_j = dataset.loc[idx_j]
                    ps_j = power_spectra.loc[idx_j]

                    similarity_data[similarities.SimilarityKeys.INDEX_PAIR].append({idx_i, idx_j})

                    # Calculate the difference between power spectra
                    for attr_name in power_spectrum_attrs:
                        similarity_data[attr_name].append(
                            utils.rmse(ps_i[attr_name], ps_j[attr_name]).cpu().item()
                        )

                    dist_trace = abs(
                        (
                            diag_mean(row_i[keys.ATOM_1_OCCS_1])
                            + diag_mean(row_i[keys.ATOM_1_OCCS_2])
                        )
                        - (
                            diag_mean(row_j[keys.ATOM_1_OCCS_1])
                            + diag_mean(row_j[keys.ATOM_1_OCCS_2])
                        )
                    )
                    similarity_data[similarities.SimilarityKeys.DIST_TRACE].append(dist_trace)

                    # Difference in input parameter values
                    similarity_data[similarities.SimilarityKeys.DELTA_PARAM_IN].append(
                        abs(row_i[keys.PARAM_IN] - row_j[keys.PARAM_IN])
                    )

        return pd.DataFrame(similarity_data)

    def identify_duplicates(
        self, dataset: pd.DataFrame, group_by=DEFAULT_GROUP_BY, tolerances=None
    ) -> pd.DataFrame:
        group_by = group_by or self.DEFAULT_GROUP_BY
        tolerances = tolerances if tolerances is not None else {}
        occs_tol = tolerances.get("occs_tol", self.OCCS_TOL)
        param_tol = tolerances.get("param_tol", self.PARAM_TOL)

        tolerances = {
            "occs_sum": occs_tol,
            "occs_prod": occs_tol,
            similarities.SimilarityKeys.DELTA_PARAM_IN: param_tol,
        }
        similarities_frame = self.get_similarity_frame(dataset, group_by=group_by)
        return similarities.identify_duplicates(dataset, similarities_frame, tolerances=tolerances)


class VGraph(e3psi.TwoSite, ModelGraph):
    """A two-site model for Hubbard V interactions"""

    uuid.UUID("be7f3ec5-7412-4a98-a71a-2ccf16e27dd1")

    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"
    DEFAULT_GROUP_BY = P_ELEMENT, D_ELEMENT

    OCCS_TOL = DEFAULT_OCCS_TOL
    PARAM_TOL = DEFAULT_PARAM_TOL
    DIST_TOL = 1e-1

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
        _check_shape(row, key("occs", pidx, 1), (3, 3))
        _check_shape(row, key("occs", pidx, 2), (3, 3))

        _check_shape(row, key("occs", didx, 1), (5, 5))
        _check_shape(row, key("occs", didx, 2), (5, 5))

        psite_inputs = self.site1.create_inputs(
            row[key("element", pidx)],
            *datasets.get_occupation_matrices(row, pidx),
            dtype=dtype,
            device=device,
        )
        psite_tensor = torch.hstack(
            tuple(
                # Use tensorial_attrs() so that the order is guaranteed to be correct
                psite_inputs[name]
                for name in e3psi.tensorial_attrs(self.site1).keys()
            )
        )

        dsite_inputs = self.site2.create_inputs(
            row[key("element", didx)],
            *datasets.get_occupation_matrices(row, didx),
            dtype=dtype,
            device=device,
        )
        dsite_tensor = torch.hstack(
            tuple(
                # Use tensorial_attrs() so that the order is guaranteed to be correct
                dsite_inputs[name]
                for name in e3psi.tensorial_attrs(self.site2).keys()
            )
        )

        edge_tensor = e3psi.create_tensor(
            self.edge, dict(v=row[keys.PARAM_IN], dist=row[keys.DIST_IN]), **kwargs
        )

        return dict(
            site1=psite_tensor,
            site2=dsite_tensor,
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

        df[keys.SPECIES] = df.apply(
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

    def get_similarity_frame(self, dataset: pd.DataFrame, group_by=DEFAULT_GROUP_BY):
        power_spectrum_attrs = ("occs_sum", "occs_prod")
        dataset = dataset.copy()
        dataset[keys.SC_PATHS] = dataset.apply(
            lambda row: str(pathlib.Path(row[keys.DIR]).parent), axis=1
        )

        def tmp_input_creator(row: Mapping, site_idx: int):
            site_name = f"site{site_idx}"
            site: Site = getattr(self, site_name)
            atom_idx = self.get_pd_idx(row)[site_idx - 1]
            return site.create_inputs(
                row[key("element", atom_idx)], *datasets.get_occupation_matrices(row, atom_idx)
            )

        for site_idx in (1, 2):
            inputs = dataset.apply(
                lambda row: tmp_input_creator(row, site_idx), axis=1, result_type="reduce"
            )

            site = getattr(self, f"site{site_idx}")
            for attr_name in power_spectrum_attrs:
                # Create the power spectrum operations that we will use for calculating similarity
                attr = e3psi.tensorial_attrs(site)[attr_name]
                calc_ps = e3psi.distances.PowerSpectrumDistance(e3psi.irreps(attr))

                ps_key = key(attr_name, site_idx)
                dataset[ps_key] = inputs.apply(lambda row: calc_ps.power_spectrum(row[attr_name]))

        similarity_data = collections.defaultdict(list)

        for _name, indexes in dataset.groupby(list(group_by)).groups.items():
            for i, idx_i in enumerate(indexes):
                row_i = dataset.loc[idx_i]

                for j_, idx_j in enumerate(indexes[i + 1 :]):
                    row_j = dataset.loc[idx_j]
                    similarity_data[similarities.SimilarityKeys.INDEX_PAIR].append({idx_i, idx_j})

                    # Calculate the difference between power spectra
                    for site_idx in (1, 2):
                        for attr_name in power_spectrum_attrs:
                            ps_key = key(attr_name, site_idx)
                            ps_i = row_i[ps_key]
                            ps_j = row_j[ps_key]
                            similarity_data[ps_key].append(utils.rmse(ps_i, ps_j).cpu().item())

                    # Difference in input parameter values
                    similarity_data[similarities.SimilarityKeys.DELTA_PARAM_IN].append(
                        abs(row_i[keys.PARAM_IN] - row_j[keys.PARAM_IN])
                    )

                    similarity_data["delta_dist"].append(
                        abs(row_i[keys.DIST_IN] - row_j[keys.DIST_IN])
                    )

        return pd.DataFrame(similarity_data)

    def identify_duplicates(
        self, dataset: pd.DataFrame, group_by=DEFAULT_GROUP_BY, tolerances=None
    ) -> pd.DataFrame:
        group_by = group_by or self.DEFAULT_GROUP_BY
        tolerances = tolerances or {}
        # Generate tolerances
        occs_tol = tolerances.get("occs_tol", self.OCCS_TOL)
        param_tol = tolerances.get("param_tol", self.PARAM_TOL)
        dist_tol = tolerances.get("dist_tol", self.DIST_TOL)

        tolerances = {
            key("occs_sum", 1): occs_tol,
            key("occs_prod", 1): occs_tol,
            key("occs_sum", 2): occs_tol,
            key("occs_prod", 2): occs_tol,
            "delta_dist": dist_tol,
            similarities.SimilarityKeys.DELTA_PARAM_IN: param_tol,
        }
        similarities_frame = self.get_similarity_frame(dataset, group_by=group_by)
        return similarities.identify_duplicates(dataset, similarities_frame, tolerances=tolerances)


class HubbardDataset(torch.utils.data.Dataset):
    """PyTorch dataset from Hubbard dataframe"""

    def __init__(
        self,
        graph: ModelGraph,
        dataframe: pd.DataFrame,
        target_column=keys.PARAM_OUT,
        dtype=None,
        device=None,
    ):
        self._graph = graph
        self._df = dataframe
        self._target_column = target_column
        self._dtype = dtype or torch.get_default_dtype()
        self._device = device
        self._cache = {}

    @property
    def graph(self) -> ModelGraph:
        return self._graph

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, item):
        try:
            return self._cache[item]
        except KeyError:
            pass

        row = self._df.iloc[item]
        inp = self._graph.create_inputs(row, dtype=self._dtype, device=self._device)
        label = torch.tensor([row[self._target_column]], dtype=self._dtype, device=self._device)
        self._cache[item] = inp, label

        return inp, label

    def all(self):
        """Get a single tensor containing all the inputs.
        Warning: if there is a lot of data it may not fit in memory"""
        return torch.utils.data.default_collate(list(self))

    def all_inputs(self):
        return torch.utils.data.default_collate(list(map(operator.itemgetter(0), self)))

    def all_outputs(self):
        return torch.utils.data.default_collate(list(map(operator.itemgetter(1), self)))


GRAPHS = {
    U: UGraph,
    V: VGraph,
}
