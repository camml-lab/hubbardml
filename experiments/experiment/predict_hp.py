import logging

import pandas as pd

import hubbardml
from hubbardml import datasets
from hubbardml import keys

_LOGGER = logging.getLogger(__name__)


def prepare_data(
    graph_data: hubbardml.GraphData,
    param_cutoff: float,
    split_fraction: float,
    group_by=None,
    duplicate_tolerances: dict = None,  # None means 'use defaults'
) -> pd.DataFrame:
    dataset = graph_data.dataset

    if param_cutoff is not None:
        # Filter out those that are below the parameter cutoff
        before = len(dataset)
        dataset = dataset[dataset[keys.PARAM_OUT] > param_cutoff]
        _LOGGER.info(
            "Removed entries with %s less then %f: before - %i after %i",
            keys.PARAM_OUT,
            param_cutoff,
            before,
            len(dataset),
        )

    if duplicate_tolerances != {}:
        dataset = graph_data.identify_duplicates(
            dataset,
            group_by=group_by,
            tolerances=duplicate_tolerances,
        )

    # Now create the test/validate split
    # return datasets.split(
    #     dataset.copy(),
    #     method="category",
    #     frac=split_fraction,
    #     category=group_by,
    #     ignore_already_labelled=False
    # )
    split_data = datasets.split_by_cluster(
        dataset.copy(), frac=split_fraction, category=group_by, ignore_already_labelled=True
    )

    return split_data
