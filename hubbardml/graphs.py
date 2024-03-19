import abc
import collections
import functools
import logging
import operator
import pathlib
import time
from typing import Iterable, Dict, List, Union, Tuple, Mapping, Any, Optional
import uuid

from e3nn import o3
import e3psi
import numpy as np
import pandas as pd
from scipy.spatial import distance
import torch
import torch.utils.data

from . import datasets
from . import keys
from . import plots
from . import similarities
from . import sites
from . import utils
from . import qe

__all__ = "ModelGraph", "UGraph", "VGraph", "HubbardDataset"

_LOGGER = logging.getLogger(__name__)

U = "U"
V = "V"

DEFAULT_OCCS_TOL = 1e-4
DEFAULT_PARAM_TOL = 1e-3  # Input Hubbard parameters less than this are considered to be identical


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

    def __eq__(self, other: Union[Any, "TensorSum"]) -> bool:
        if not isinstance(other, TensorSum):
            return False

        return self._attr == other._attr


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

    def __init__(self, species: Iterable[str], occ_irrep: Union[str, o3.Irrep]) -> None:
        super().__init__()
        self.specie = e3psi.SpecieOneHot(species)
        self._occ_irrep = o3.Irrep(occ_irrep)
        self._occs = qe.QeOccuMtx(occ_irrep)  # The occupations matrix representation
        # self._occs = e3psi.OccuMtx(occ_irrep)  # The occupations matrix representation

        self.occs_sum = TensorSum(self._occs)
        self.occs_prod = TensorElementwiseProduct(self._occs)  # , filter_ir_out=["0e", "2e"])

    def create_inputs(self, tensors: Mapping, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        occupations = [tensors["occs1"], tensors["occs2"]]

        # We have to take the absolute value of the occupation matrices here because otherwise we won't
        # be globally invariant to spin flips
        # occupations = list(map(lambda occs: np.abs(np.array(occs)), occupations))

        # occupations = [self.qe_to_e3(tensors["occs1"]), self.qe_to_e3(tensors["occs2"])]

        tensor_kwargs = dict(dtype=dtype, device=device)
        return dict(
            specie=e3psi.create_tensor(self.specie, tensors["specie"], **tensor_kwargs),
            # Pass the same up/down occupation matrices to both the sum and product
            occs_sum=e3psi.create_tensor(self.occs_sum, occupations, **tensor_kwargs),
            occs_prod=e3psi.create_tensor(self.occs_prod, occupations, **tensor_kwargs),
        )

    def qe_to_e3(self, occu_mtx: torch.Tensor) -> torch.Tensor:
        """Convert from QE to e3nn convention for spherical harmonics"""
        occu_mtx = torch.tensor(occu_mtx, dtype=torch.get_default_dtype())
        cob = qe.qe_to_e3_cob(self._occ_irrep.l)  # Get the change of basis matrix
        return cob.T @ occu_mtx @ cob
        # return cob @ occu_mtx @ cob.T


class USite(Site):
    def __init__(
        self, species: Iterable[str], occ_irreps: Union[str, o3.Irrep], with_param_in=True
    ):
        super().__init__(species, occ_irreps)
        if with_param_in:
            self.u_in = o3.Irrep("0e")  # The input Hubbard parameter

    def create_inputs(self, tensors: Mapping, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        inputs = super().create_inputs(tensors, dtype=dtype, device=device)
        if getattr(self, "u_in", None) is not None:
            inputs["u_in"] = e3psi.create_tensor(
                self.u_in, tensors["u_in"], dtype=dtype, device=device
            )
        return inputs


class ModelGraph(metaclass=abc.ABCMeta):
    DEFAULT_GROUP_BY = None  # Subclasses can overwrite this to something that makes sense for them

    @abc.abstractmethod
    def create_inputs(self, raw_data, dtype=None, device=None) -> Dict:
        """Create the inputs for a model"""

    @abc.abstractmethod
    def prepare_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process and filter and the dataset to prepare it for use by this graph"""

    def calculate_duplicates_data(self, dataset: pd.DataFrame):
        """Create data needed to identify duplicates"""

    def get_similarity_frame(self, dataset: pd.DataFrame, group_by=None) -> pd.DataFrame:
        """Create a data frame containing similarities information"""

    def identify_duplicates(
        self,
        dataset: pd.DataFrame,
        group_by: Optional[Iterable[str]],
        tolerances: dict = None,
    ) -> pd.DataFrame:
        """Identify duplicate data in a dataset"""
        if group_by is None:
            group_by = self.DEFAULT_GROUP_BY

        similarities_frame = self.get_similarity_frame(dataset, group_by=group_by)
        return self.identify_duplicates_(dataset, similarities_frame, tolerances=tolerances)

    @abc.abstractmethod
    def identify_duplicates_(
        self,
        dataset: pd.DataFrame,
        similarities_frame: pd.DataFrame,
        tolerances: dict = None,
    ) -> pd.DataFrame:
        """Identify duplicate data in a dataset using the given similarities frame"""


class UGraph(e3psi.graphs.OneSite, ModelGraph):
    """A graph that contains only one (d-element) site"""

    TYPE_ID = uuid.UUID("4d7951c8-5fc7-4c9d-883e-ef09d27f478c")

    DEFAULT_GROUP_BY = (keys.ATOM_1_ELEMENT,)
    OCCS_TOL = DEFAULT_OCCS_TOL
    PARAM_TOL = DEFAULT_PARAM_TOL

    def __init__(self, species: Iterable[str], with_param_in=True) -> None:
        self.species = tuple(species)
        dsite = USite(species, "2e", with_param_in=with_param_in)
        super().__init__(dsite)

    def create_inputs(self, row: Mapping, dtype=None, device=None) -> Dict:
        """Create a tensor from a dataframe row or dictionary"""
        site_inputs = self.site.create_inputs(
            dict(
                specie=row[keys.ATOM_1_ELEMENT],
                u_in=row.get(keys.PARAM_IN, None),
                **datasets.get_occupation_matrices(row, 1),
            ),
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
        # Do some initial filtering
        df = datasets.filter_dataset(
            df,
            param_type=keys.PARAM_U,
            remove_vwd=True,
            remove_zero_out=True,
            remove_in_eq_out=False,
        )

        # Remove non rows corresponding to unsupported species
        df = df[df[keys.ATOM_1_ELEMENT].isin(self.species)]
        df = df[df[keys.ATOM_2_ELEMENT].isin(self.species)]

        # Remove rows that we can't deal with because they do not have the right electronic configuration
        for col in (keys.ATOM_1_OCCS_1, keys.ATOM_1_OCCS_2, keys.ATOM_2_OCCS_1, keys.ATOM_2_OCCS_2):
            occs_filter = df[col].apply(len) != 5  # 5x5 occupation matrices (for d-elements)
            if sum(occs_filter) > 0:
                _LOGGER.warning(
                    "Found %n occupation matrices of the wrong shape in column %s.  Removing.",
                    sum(occs_filter),
                    col,
                )
                df = df[~occs_filter]

        # Add additional useful columns
        df[keys.SPECIES] = df.apply(
            lambda row: frozenset([row[keys.ATOM_1_ELEMENT], row[keys.ATOM_2_ELEMENT]]), axis=1
        )
        df[keys.LABEL] = df[keys.ATOM_1_ELEMENT]
        df[keys.COLOUR] = df[keys.ATOM_1_ELEMENT].map(plots.element_colours)
        df = _prepare_dataset(df)

        return df

    def calculate_duplicates_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Create data needed to identify duplicates"""
        power_spectrum_attrs = ("occs_sum", "occs_prod")
        dataset = _prepare_dataset(dataset)

        with torch.no_grad():
            # Create the power spectrum operations that we will use for calculating similarity
            inputs = dataset.apply(
                lambda row: self.site.create_inputs(
                    dict(
                        specie=row[keys.ATOM_1_ELEMENT],
                        u_in=row[keys.PARAM_IN],
                        **datasets.get_occupation_matrices(row, 1),
                    )
                ),
                axis=1,
                result_type="reduce",
            )
            attrs = e3psi.tensorial_attrs(self.site)
            for attr_name in power_spectrum_attrs:
                # Get the attribute and create the power spectrum operation for it
                attr = attrs[attr_name]
                dist_ps = e3psi.distances.PowerSpectrumDistance(e3psi.irreps(attr))

                ps_key = key(attr_name, 1)

                # Calculate The power spectra
                dataset[ps_key] = inputs.apply(
                    lambda row: dist_ps.power_spectrum(row[attr_name]).numpy()
                )

        # OCCS_TRACE
        dataset[key("occs_trace", 1)] = dataset.apply(
            lambda row: diag_mean(row[key("occs", 1, 1)]) + diag_mean(row[key("occs", 1, 2)]),
            axis=1,
        )

        return dataset

    def get_similarity_frame(self, dataset: pd.DataFrame, group_by=DEFAULT_GROUP_BY):
        power_spectrum_attrs = ("occs_sum", "occs_prod")
        dataset = _prepare_dataset(dataset)

        dataset = self.calculate_duplicates_data(dataset)

        sim_data = collections.defaultdict(list)
        for _name, group in dataset.groupby(list(group_by)):
            row_ijs = np.fromiter(utils.linear_index_pair(len(group)), dtype=((int, 2)))
            # These contain the index value for each pair of rows (in the same order as returned by pdist)
            indexes = [set(entry) for entry in group.index.to_numpy()[row_ijs]]
            sim_data[similarities.SimilarityKeys.INDEX_PAIR].extend(indexes)

            # PARAM_IN
            array = np.vstack(group[keys.PARAM_IN].to_numpy())
            dists = distance.pdist(array)
            sim_data[similarities.SimilarityKeys.DELTA_PARAM_IN].extend(dists)

            # Calculate the difference between power spectra
            for attr_name in power_spectrum_attrs:
                ps_key = key(attr_name, 1)
                array = np.vstack(group[ps_key].to_numpy())
                dists = np.sqrt(distance.pdist(array, "sqeuclidean") / len(array[0]))
                sim_data[attr_name].extend(dists)

            # OCCS_TRACE
            array = np.vstack(group[key("occs_trace", 1)])
            dists = distance.pdist(array)
            sim_data[similarities.SimilarityKeys.DIST_TRACE].extend(dists)

        sim_frame = pd.DataFrame(sim_data)
        return sim_frame

    def identify_duplicates_(
        self, dataset: pd.DataFrame, similarities_frame: pd.DataFrame, tolerances=None
    ) -> pd.DataFrame:
        tolerances = tolerances if tolerances is not None else {}
        occs_tol = tolerances.get("occs_tol", self.OCCS_TOL)
        param_tol = tolerances.get("param_tol", self.PARAM_TOL)

        tolerances = {
            "occs_sum": occs_tol,
            "occs_prod": occs_tol,
            similarities.SimilarityKeys.DELTA_PARAM_IN: param_tol,
        }
        return similarities.identify_duplicates(dataset, similarities_frame, tolerances=tolerances)


class VGraph(e3psi.TwoSite, ModelGraph):
    """A two-site model for Hubbard V interactions"""

    uuid.UUID("be7f3ec5-7412-4a98-a71a-2ccf16e27dd1")

    P_ELEMENT = "p_element"
    D_ELEMENT = "d_element"
    DEFAULT_GROUP_BY = P_ELEMENT, D_ELEMENT

    OCCS_TOL = DEFAULT_OCCS_TOL
    PARAM_TOL = 5e-3
    DIST_TOL = 4e-3

    def __init__(self, species: Iterable[str], with_param_in=True):
        # Create the graph by supplying the sites
        self._species = tuple(species)
        psite = Site(species, "1o")
        dsite = Site(species, "2e")
        super().__init__(psite, dsite, sites.VEdge(with_param_in=with_param_in))

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
            dict(
                specie=row[key("element", pidx)],
                **datasets.get_occupation_matrices(row, pidx),
            ),
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
            dict(
                specie=row[key("element", didx)],
                **datasets.get_occupation_matrices(row, didx),
            ),
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
            self.edge, dict(v=row.get(keys.PARAM_IN, None), dist=row[keys.DIST_IN]), **kwargs
        )

        return dict(
            site1=psite_tensor,
            site2=dsite_tensor,
            edge=edge_tensor,
        )

    def prepare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = datasets.filter_dataset(
            df,
            param_type=keys.PARAM_V,
            remove_vwd=True,
            remove_zero_out=False,
            remove_in_eq_out=False,
        )
        # Remove non rows corresponding to unsupported species
        df = df[df[keys.ATOM_1_ELEMENT].isin(self.species)]
        df = df[df[keys.ATOM_2_ELEMENT].isin(self.species)]

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
        df[self.P_ELEMENT] = df.apply(
            lambda row: (
                row[keys.ATOM_1_ELEMENT]
                if row[keys.ATOM_1_OCCS_1].shape[0] == 3
                else row[keys.ATOM_2_ELEMENT]
            ),
            axis=1,
        )
        df[self.D_ELEMENT] = df.apply(
            lambda row: (
                row[keys.ATOM_1_ELEMENT]
                if row[keys.ATOM_2_OCCS_1].shape[0] == 3
                else row[keys.ATOM_2_ELEMENT]
            ),
            axis=1,
        )
        df[keys.LABEL] = df.apply(
            lambda row: f"{row[self.D_ELEMENT]}-{row[self.P_ELEMENT]}", axis=1
        )
        df[keys.COLOUR] = df[self.D_ELEMENT].map(plots.element_colours)
        df = _prepare_dataset(df)

        return df

    def calculate_duplicates_data(self, dataset: pd.DataFrame):
        """Create data needed to identify duplicates"""
        _LOGGER.info("Calculating duplicates data")
        with torch.no_grad():
            power_spectrum_attrs = ("occs_sum", "occs_prod")
            dataset = _prepare_dataset(dataset)

            def tmp_input_creator(row: Mapping, site_idx: int):
                site_name = f"site{site_idx}"
                site: Site = getattr(self, site_name)
                atom_idx = self.get_pd_idx(row)[site_idx - 1]
                return site.create_inputs(
                    dict(
                        specie=row[key("element", atom_idx)],
                        **datasets.get_occupation_matrices(row, atom_idx),
                    )
                )

            _LOGGER.info("Creating power spectra for %i input tensors", len(dataset))
            for atom_idx in (1, 2):
                _LOGGER.info("Starting site %i", atom_idx)
                inputs = dataset.apply(
                    lambda row: tmp_input_creator(row, atom_idx), axis=1, result_type="reduce"
                )

                site = getattr(self, f"site{atom_idx}")
                for attr_name in power_spectrum_attrs:
                    _LOGGER.info("Getting power spectra for %s", attr_name)
                    # Create the power spectrum operations that we will use for calculating similarity
                    attr = e3psi.tensorial_attrs(site)[attr_name]
                    calc_ps = e3psi.distances.PowerSpectrumDistance(e3psi.irreps(attr))

                    ps_key = key(attr_name, atom_idx)
                    # dataset[ps_key] = inputs[attr_name].apply(calc_ps.power_spectrum)

                    dataset[ps_key] = inputs.apply(
                        lambda row: calc_ps.power_spectrum(row[attr_name])
                    )
                _LOGGER.info("Finished site %i", atom_idx)

                # OCCS_TRACE
                dataset[key("occs_trace", atom_idx)] = dataset.apply(
                    lambda row: diag_mean(row[key("occs", atom_idx, 1)])
                    + diag_mean(row[key("occs", atom_idx, 2)]),
                    axis=1,
                )

        return dataset

    def get_similarity_frame(self, dataset: pd.DataFrame, group_by=DEFAULT_GROUP_BY):
        _LOGGER.info("Creating similarity frame grouped by: %s", group_by)
        dataset = self.calculate_duplicates_data(dataset)

        power_spectrum_attrs = ("occs_sum", "occs_prod")

        _LOGGER.info("Comparing power spectra")
        sim_data = collections.defaultdict(list)
        for _name, group in dataset.groupby(list(group_by)):
            _LOGGER.info("Comparing %i entries", len(group))

            row_ijs = np.fromiter(utils.linear_index_pair(len(group)), dtype=((int, 2)))
            # These contain the index value for each pair of rows (in the same order as returned by pdist)
            indexes = [set(entry) for entry in group.index.to_numpy()[row_ijs]]
            sim_data[similarities.SimilarityKeys.INDEX_PAIR].extend(indexes)

            # PARAM_IN
            array = np.vstack(group[keys.PARAM_IN].to_numpy())
            dists = distance.pdist(array)
            sim_data[similarities.SimilarityKeys.DELTA_PARAM_IN].extend(dists)

            # DIST IN
            array = np.vstack(group[keys.DIST_IN].to_numpy())
            dists = distance.pdist(array)
            sim_data["delta_dist"].extend(dists)

            # Calculate the difference between power spectra
            for site_idx in (1, 2):
                for attr_name in power_spectrum_attrs:
                    ps_key = key(attr_name, site_idx)
                    array = np.vstack(group[ps_key].to_numpy())
                    dists = np.sqrt(distance.pdist(array, "sqeuclidean") / len(array[0]))
                    sim_data[attr_name].extend(dists)

        return pd.DataFrame(sim_data)

    def get_similarity_frame_old(self, dataset: pd.DataFrame, group_by=DEFAULT_GROUP_BY):
        _LOGGER.info("Creating similarity frame grouped by: %s", group_by)
        power_spectrum_attrs = ("occs_sum", "occs_prod")
        dataset = self.calculate_duplicates_data(dataset)

        _LOGGER.info("Comparing power spectra")
        similarity_data = collections.defaultdict(list)
        for _name, indexes in dataset.groupby(list(group_by)).groups.items():
            _LOGGER.info("Comparing %i groups", len(indexes))
            for i, idx_i in enumerate(indexes):
                row_i = dataset.loc[idx_i]

                for _j, idx_j in enumerate(indexes[i + 1 :]):
                    row_j = dataset.loc[idx_j]
                    similarity_data[similarities.SimilarityKeys.INDEX_PAIR].append({idx_i, idx_j})

                    # Calculate the difference between power spectra
                    for site_idx in (1, 2):
                        for attr_name in power_spectrum_attrs:
                            ps_key = key(attr_name, site_idx)
                            ps_i = row_i[ps_key]
                            ps_j = row_j[ps_key]
                            similarity_data[ps_key].append(utils.rmse(ps_i, ps_j))

                    # Difference in input parameter values
                    similarity_data[similarities.SimilarityKeys.DELTA_PARAM_IN].append(
                        abs(row_i[keys.PARAM_IN] - row_j[keys.PARAM_IN])
                    )

                    similarity_data["delta_dist"].append(
                        abs(row_i[keys.DIST_IN] - row_j[keys.DIST_IN])
                    )

        return pd.DataFrame(similarity_data)

    def identify_duplicates_(
        self, dataset: pd.DataFrame, similarities_frame: pd.DataFrame, tolerances=None
    ) -> pd.DataFrame:
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
        label = torch.as_tensor([row[self._target_column]], dtype=self._dtype, device=self._device)
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
