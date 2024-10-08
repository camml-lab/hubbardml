import logging
import pathlib

import pandas as pd

import hubbardml
from hubbardml import datasets
from hubbardml import keys
from hubbardml import plots

_LOGGER = logging.getLogger(__name__)


def prepare_data(
    graph_data: hubbardml.GraphData,
    param_cutoff: float,
    duplicate_tolerances: dict = None,  # None means 'use defaults'
    uv_iter: int = 2,
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

    dataset = set_training_labels(dataset, uv_iter=uv_iter, include_subsequent=False)

    # This will set the training label to DUPLICATE for all but one entry in each cluster of identical inputs
    dups = graph_data.identify_duplicates(
        dataset[dataset[keys.TRAINING_LABEL] == keys.VALIDATE],
        group_by=[keys.SPECIES, keys.SC_PATHS],
        tolerances=duplicate_tolerances,
    )
    # Copy over the duplicates label to our set
    dataset.loc[dups[dups[keys.TRAINING_LABEL] == keys.DUPLICATE].index, keys.TRAINING_LABEL] = (
        keys.DUPLICATE
    )

    return dataset


def set_training_labels(
    data: pd.DataFrame,
    uv_iter: int,
    include_subsequent=False,
) -> pd.DataFrame:
    """
    Label all data with uv_iter less than the passed value as training, and data at uv_iter (and higher if
    `include_subsequent`=True) as validation.  All the rest of the rows will have their training label set to `None`.
    """
    _LOGGER.info(
        "Setting uv_iter %i (include_subsequent=%s) as validation, lower as training",
        uv_iter,
        include_subsequent,
    )
    # Exclude all rows from the experiment, and then manually activate validation and training for those we want
    data[keys.TRAINING_LABEL] = None

    # Train on all previous iterations
    data.loc[data[keys.UV_ITER] < uv_iter, keys.TRAINING_LABEL] = keys.TRAIN

    for path, sc_rows in datasets.iter_self_consistent_paths(data):
        for pair, rows in datasets.iter_atom_idx_pairs(sc_rows):
            min_iter = rows[keys.UV_ITER].min()
            if min_iter < uv_iter:
                # Set the current and all future iterations as validation
                if include_subsequent:
                    next_iters = rows[rows[keys.UV_ITER] >= uv_iter]
                else:
                    next_iters = rows[rows[keys.UV_ITER] == uv_iter]

                data.loc[next_iters.index, keys.TRAINING_LABEL] = keys.VALIDATE

                # Find the data from the previous iteration that has a result
                previous_iter = rows.loc[rows[keys.UV_ITER] < uv_iter, keys.UV_ITER].max()
                previous_iter_row = rows[rows[keys.UV_ITER] == previous_iter]

                param_in = previous_iter_row[keys.PARAM_OUT].item()
                data.loc[next_iters.index, keys.PARAM_OUT_PREDICTED] = param_in

    return data


def plot_baseline(dataset: pd.DataFrame, output_dir: pathlib.Path):
    """Make a plot showing the baseline parity results"""
    dataset = dataset.copy()

    # Train only on the first iteration
    # for path, sc_rows in datasets.iter_self_consistent_paths(dataset):
    #     for pair, rows in datasets.iter_atom_idx_pairs(sc_rows):
    #         min_iter = rows[keys.UV_ITER].min()
    #         first_iter = rows[rows[keys.UV_ITER] == min_iter]
    #         assert len(first_iter) == 1  # nosec
    #
    #         # Train from first iteration
    #         dataset.loc[sc_rows.index, keys.PARAM_OUT_PREDICTED] = first_iter.iloc[0][
    #             keys.PARAM_OUT
    #         ]

    dataset = dataset[dataset[keys.TRAINING_LABEL] == keys.VALIDATE]
    fig = plots.split_plot(
        dataset,
        keys.LABEL,
        axis_label="Hubbard param. (eV)",
        title=f"Baseline model, RMSE = {datasets.rmse(dataset):.2f} eV",
    )

    plot_path = output_dir / pathlib.Path("plots/")
    plot_path.mkdir(exist_ok=True)

    fig.savefig(plot_path / "parity_baseline.pdf", bbox_inches="tight")
