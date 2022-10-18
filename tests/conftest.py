# -*- coding: utf-8 -*-
# pylint: disable=unused-import, redefined-outer-name
import pathlib
import pytest

from hubbardml import datasets
from hubbardml import keys


@pytest.fixture
def dataframe():  # noqa: F811
    df = datasets.load(pathlib.Path(__file__).parent.parent / "data" / "data_uv_2022_8_18.json")
    # Split into test/train
    datasets.split(df)

    df = df[~(df[keys.PARAM_IN] == df[keys.PARAM_OUT])]
    # df = df[~((df[keys.ATOM_1_ELEMENT] != 'O') & (df[keys.ATOM_2_ELEMENT] != 'O'))]

    df = df[~(df[keys.ATOM_1_ELEMENT] == df[keys.ATOM_2_ELEMENT])]

    for elem in "O", "Ni", "Co":
        df = df[~((df[keys.ATOM_1_ELEMENT] == elem) & (df[keys.ATOM_2_ELEMENT] == elem))]

    return df
