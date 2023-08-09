import pathlib
import logging

import pandas as pd

from hubbardml import graphs
from hubbardml import datasets
from hubbardml import keys

_LOGGER = logging.getLogger(__name__)


def prepare_data(
    graph: graphs.ModelGraph,
    dataset: pd.DataFrame,
    output_dir: pathlib.Path,
    param_cutoff: float,
    training_split: float,
    group_by=None,
    duplicate_tolerances=None,
) -> pd.DataFrame:
    # Filter out anything smaller than the cutoff
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

    dataset = datasets.generate_converged_prediction_dataset(dataset)

    if duplicate_tolerances != {}:
        dataset = graph.identify_duplicates(
            dataset, group_by=group_by, tolerances=duplicate_tolerances
        )

    # For now, just copy over the final parameter to be used as the label
    dataset[keys.PARAM_OUT] = dataset[keys.PARAM_OUT_FINAL]

    return datasets.split_by_cluster(
        dataset.copy(), frac=training_split, category=group_by, ignore_already_labelled=True
    )
