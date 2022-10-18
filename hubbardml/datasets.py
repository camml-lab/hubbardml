import json
import pathlib
from typing import List, Union, Tuple, Set

import ase.data
import numpy as np
import pandas as pd

from . import keys
from . import utils

# A cutoff below which Hubbard corrections are not applied, even if HP calculates the output Hubbard params for
# all Hubbard-active species.  This is a threshold set by us and not in practice during self-consistent Hubbard
# calculations where the Hubbard-active sites are typically set manually based on knowledge of the neighbour
# environment
HUBBARD_CUTOFF = 0.25


def split(dataset: pd.DataFrame, frac=0.2, method="simple", **kwargs) -> pd.DataFrame:
    if method == "simple":
        return split_simple(dataset, frac)
    elif method == "category":
        return split_within_category(dataset, frac=frac, **kwargs)
    else:
        raise ValueError(f"Unknown randomisation method: {method}")


def split_simple(dataset: pd.DataFrame, frac=0.2) -> pd.DataFrame:
    """Randomise a dataset by treating each row as independent"""

    dataset[keys.TRAINING_LABEL] = keys.TRAIN
    dataset.loc[dataset.sample(frac=frac).index, keys.TRAINING_LABEL] = keys.TEST

    return dataset


def split_within_category(dataset: pd.DataFrame, frac=0.2, category: Union[str, List] = None) -> pd.DataFrame:
    """Randomise a fraction within each category"""
    dataset[keys.TRAINING_LABEL] = keys.TRAIN

    def set_test(frame):
        dataset.loc[frame.sample(frac=frac).index, keys.TRAINING_LABEL] = keys.TEST

    # For test we only use
    subset = dataset[dataset[keys.PARAM_OUT] > HUBBARD_CUTOFF]
    subset.groupby(category).apply(set_test)

    return dataset


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a DFT+Hubbard dataframe, generating any necessary derivative data and labels"""

    # Convert occupation matrices to numpy arrays
    for occup_label in (
        keys.ATOM_1_OCCS_1,
        keys.ATOM_2_OCCS_1,
        keys.ATOM_1_OCCS_2,
        keys.ATOM_2_OCCS_2,
    ):
        df[occup_label] = df[occup_label].apply(_create_array)

        # Check  that the occupation matrices are symmetric as expected
        if sum(df[occup_label].apply(symm_test) == False) != 0:  # noqa: E712
            raise ValueError(f"Occupations matrices didn't all pass symmetry test {occup_label}")

    # Now calculate permutationally invariant features
    df[keys.ATOM_1_OCCS_INV_1] = df.apply(lambda row: row[keys.ATOM_1_OCCS_1] + row[keys.ATOM_1_OCCS_2], axis=1)
    df[keys.ATOM_1_OCCS_INV_2] = df.apply(
        lambda row: np.multiply(row[keys.ATOM_1_OCCS_1], row[keys.ATOM_1_OCCS_2]), axis=1
    )

    df[keys.ATOM_2_OCCS_INV_1] = df.apply(lambda row: row[keys.ATOM_2_OCCS_1] + row[keys.ATOM_2_OCCS_2], axis=1)
    df[keys.ATOM_2_OCCS_INV_2] = df.apply(
        lambda row: np.multiply(row[keys.ATOM_2_OCCS_1], row[keys.ATOM_2_OCCS_2]), axis=1
    )

    mask = df.apply(lambda row: row[keys.ATOM_2_IDX] > row[keys.N_ATOM_UC], axis=1)
    df.drop(df[mask].index)

    # for directory in df[keys.DIR].unique():
    #     new_rows = df[
    #         (df[keys.PARAM_TYPE] == keys.PARAM_V) &\
    #         (df[keys.DIR] == directory) &\
    #         (df[keys.UV_ITER] == 2)
    #     ].copy()
    #     new_rows[keys.PARAM_OUT] = new_rows[keys.PARAM_IN]
    #     new_rows[keys.PARAM_IN] = 0.
    #     uvals = df[
    #         (df[keys.PARAM_TYPE] == keys.PARAM_U) &\
    #         (df[keys.DIR] == directory) &\
    #         (df[keys.UV_ITER] == 2)
    #     ]

    # Strip any whitespace from elements
    for key in (keys.ATOM_1_ELEMENT, keys.ATOM_2_ELEMENT):
        df[key] = df[key].str.strip()

    df[keys.PARAM_DELTA] = df[keys.PARAM_OUT] - df[keys.PARAM_IN]

    return df


def filter_dataset(
    df: pd.DataFrame,
    remove_vwd=True,
    remove_zero_out=True,
    remove_in_eq_out=True,
    param_type=None,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(df)

    if param_type is not None:
        # Keep only the specified parameter type
        df = df[df[keys.PARAM_TYPE] == param_type]
        if param_type == keys.PARAM_V:
            df = df[~(df[keys.ATOM_1_ELEMENT] == df[keys.ATOM_2_ELEMENT])]

    if remove_zero_out:
        # Remove any rows where the output parameter is (close to) zero
        df = df[df[keys.PARAM_OUT] >= 1e-3]

    if remove_vwd:
        # Remove any rows that correspond to runs with a van der Waals functional
        df = df[df[keys.IS_VDW] == False]  # noqa: E712

    if remove_in_eq_out:
        # Remove any rows where the input and output parameters are identical
        df = df[~(df[keys.PARAM_IN] == df[keys.PARAM_OUT])]

    return df


def generate_converged_prediction_dataset(df: pd.DataFrame):
    """This function takes a dataset, groups them by directory and sets the PARAM_OUT to the value of from the final
    iteration in that folder.  This is useful when training a model that should predict the final converged Hubbard
    parameter (rather than that of the next step in the self-consistent procedure)"""
    paths = get_self_consistent_paths(df)
    new = df.copy(deep=True)

    for path in paths:
        path_mask = df[keys.DIR].str.startswith(path)
        # Get the maximum iteration reached in that path
        max_iter = df[path_mask][keys.UV_ITER].max()

        row_mask = path_mask & (df[keys.UV_ITER] == max_iter)
        for _, row in df[row_mask].iterrows():
            # Find the path and atom indices that match
            atom_1_mask = df[keys.ATOM_1_IDX] == row[keys.ATOM_1_IDX]
            atom_2_mask = df[keys.ATOM_2_IDX] == row[keys.ATOM_2_IDX]

            # Set all iterations to have the output parameter of the final iteration (hopefully converged)
            new.loc[path_mask & atom_1_mask & atom_2_mask, keys.PARAM_OUT_FINAL] = row[keys.PARAM_OUT]

    return new


def symm_test(mtx):
    return (mtx == mtx.T).all()


def load(filename: Union[pathlib.Path, str]) -> pd.DataFrame:
    with open(filename) as file:
        json_data = json.load(file)

    df = pd.DataFrame(json_data)
    return preprocess(df)


def rmse(df: pd.DataFrame, label: str = keys.TEST) -> float:
    if label in (keys.TEST, keys.TRAIN):
        df = df[df[keys.TRAINING_LABEL] == label]
    elif label != "both":
        raise ValueError(label)

    return utils.rmse(df[keys.PARAM_OUT], df[keys.PARAM_OUT_PREDICTED])


def _create_array(occu: Union[List, np.ndarray]) -> np.ndarray:
    """Create a numpy array from a list of occupation matrix entries"""
    if isinstance(occu, np.ndarray):
        return occu

    arr = np.array(occu)

    if len(arr.shape) == 2:
        if arr.shape[0] == arr.shape[1]:
            return arr

        raise ValueError(f"Unexpected occupations matrix: {arr}")
    if len(occu) == 25:
        # D-block
        return arr.reshape(5, 5)
    elif len(occu) == 9:
        # P-block
        return arr.reshape(3, 3)
    else:
        raise ValueError(f"Unknown occupations list: {occu}")


def element_pair(row) -> Tuple:
    """From a dataframe row or dictionary get the atom element pair sorted by highest atomic number first"""
    return tuple(
        sorted(
            [row[keys.ATOM_1_ELEMENT], row[keys.ATOM_2_ELEMENT]], key=lambda x: ase.data.atomic_numbers[x], reverse=True
        )
    )


def get_self_consistent_paths(df: pd.DataFrame) -> Set[pathlib.Path]:
    """Returns a set of paths that are the root of the self-consistent Hubbard calculations"""
    return {str(pathlib.Path(directory).parent) for directory in df[keys.DIR].unique()}
