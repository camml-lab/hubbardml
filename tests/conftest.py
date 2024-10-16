# -*- coding: utf-8 -*-
# pylint: disable=unused-import, redefined-outer-name
import pathlib
from urllib import request

import pytest

from hubbardml import datasets
from hubbardml import keys

LOCAL_DATA = pathlib.Path(__file__).parent.parent / "data" / "data_uv_2024_1_25.arrow"
REMOTE_DATA = (
    "https://archive.materialscloud.org/record/file?record_id=2389&filename=data_uv_2024_1_25.arrow"
)


@pytest.fixture
def dataframe():  # noqa: F811
    if not LOCAL_DATA.exists():
        LOCAL_DATA.parent.mkdir(exist_ok=True)
        request.urlretrieve(REMOTE_DATA, LOCAL_DATA)

    df = datasets.load(LOCAL_DATA)

    # Split into test/train
    datasets.split(df)

    df = df[~(df[keys.PARAM_IN] == df[keys.PARAM_OUT])]
    # df = df[~((df[keys.ATOM_1_ELEMENT] != 'O') & (df[keys.ATOM_2_ELEMENT] != 'O'))]

    df = df[~(df[keys.ATOM_1_ELEMENT] == df[keys.ATOM_2_ELEMENT])]

    for elem in "O", "Ni", "Co":
        df = df[~((df[keys.ATOM_1_ELEMENT] == elem) & (df[keys.ATOM_2_ELEMENT] == elem))]

    return df
