"""
Experiment that tests predicting the final (self-consistent) value from any intermediate step in the self-consistent
procedure.
"""

import logging

import pandas as pd

import hubbardml
from hubbardml import datasets
from hubbardml import keys

_LOGGER = logging.getLogger(__name__)


def prepare_data(
    graph_data: hubbardml.GraphData,
    param_cutoff: float,
    training_split: float,
    group_by=None,
    duplicate_tolerances=None,
) -> pd.DataFrame:
    dataset = graph_data.dataset

    # Filter out anything smaller than the cutoff
    if param_cutoff is not None:
        # Filter out those that are below the parameter cutoff
        before = len(dataset)
        dataset.drop(dataset.index[dataset[keys.PARAM_OUT] <= param_cutoff], inplace=True)
        _LOGGER.info(
            "Removed entries with %s less then %f: before - %i after %i",
            keys.PARAM_OUT,
            param_cutoff,
            before,
            len(dataset),
        )

    dataset = datasets.generate_converged_prediction_dataset(dataset, copy=False)

    if duplicate_tolerances != {}:
        # De-duplicate
        graph_data._dataset = dataset
        _LOGGER.debug("Identifying duplicates grouped by: %s", group_by)
        graph_data.identify_duplicates(dataset, group_by=group_by, tolerances=duplicate_tolerances)
        _LOGGER.debug("Dataset size: %i", len(dataset))

    # For now, just copy over the final parameter to be used as the label
    dataset[keys.PARAM_OUT] = dataset[keys.PARAM_OUT_FINAL]

    return datasets.split_by_cluster(
        dataset.copy(), frac=training_split, category=group_by, ignore_already_labelled=True
    )
