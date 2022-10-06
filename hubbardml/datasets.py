import json
from typing import List, Union

import numpy as np
import pandas as pd

from . import keys
from . import utils


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

    dataset.groupby(category).apply(set_test)

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

    # Now calculate permutational invariant versions
    df[keys.ATOM_1_OCCS_INV_1] = df.apply(lambda row: row[keys.ATOM_1_OCCS_1] + row[keys.ATOM_1_OCCS_2], axis=1)
    df[keys.ATOM_1_OCCS_INV_2] = df.apply(
        lambda row: np.multiply(row[keys.ATOM_1_OCCS_1], row[keys.ATOM_1_OCCS_2]), axis=1
    )

    df[keys.ATOM_2_OCCS_INV_1] = df.apply(lambda row: row[keys.ATOM_2_OCCS_1] + row[keys.ATOM_2_OCCS_2], axis=1)
    df[keys.ATOM_2_OCCS_INV_2] = df.apply(
        lambda row: np.multiply(row[keys.ATOM_2_OCCS_1], row[keys.ATOM_2_OCCS_2]), axis=1
    )

    if sum(df[keys.ATOM_1_OCCS_1].apply(symm_test) == False) != 0:  # noqa: E712
        raise ValueError(f"Occupations matcies didn't all pass symmetry test {keys.ATOM_1_OCCS_1}")

    df.loc[df[keys.ATOM_1_IDX] == df[keys.ATOM_2_IDX], keys.PARAM_TYPE] == "U"
    df.loc[df[keys.ATOM_1_IDX] != df[keys.ATOM_2_IDX], keys.PARAM_TYPE] == "V"

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


def generate_v_zero(df: pd.DataFrame) -> pd.DataFrame:
    for directory in df[keys.DIR].unique():
        # Get V values in this directly
        min_iter = df[keys.UV_ITER].min()

        new_rows = df[
            (df[keys.PARAM_TYPE] == keys.PARAM_V) & (df[keys.DIR] == directory) & (df[keys.UV_ITER] == min_iter)
        ].copy()

        # Copy over the former param in as the previous iter's out
        new_rows[keys.PARAM_OUT] = new_rows[keys.PARAM_IN]
        # Zero out the param in
        new_rows[keys.PARAM_IN] = 0.0

        # Now we need to get the occupation matrices
        uvals = df[  # noqa: F841
            (df[keys.PARAM_TYPE] == keys.PARAM_U) & (df[keys.DIR] == directory) & (df[keys.UV_ITER] == 2)
        ]


def symm_test(mtx):
    return (mtx == mtx.T).all()


def load(filename: str) -> pd.DataFrame:
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
    if isinstance(occu, np.ndarray):
        return occu

    if len(occu) == 25:
        # D-block
        return np.array(occu).reshape(5, 5)
    elif len(occu) == 9:
        # P-block
        return np.array(occu).reshape(3, 3)
    else:
        raise ValueError(f"Unknown occupations list: {occu}")
