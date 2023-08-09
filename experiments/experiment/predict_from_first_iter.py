import logging
import pathlib

import pandas as pd

from hubbardml import datasets
from hubbardml import graphs
from hubbardml import keys
from hubbardml import plots

_LOGGER = logging.getLogger(__name__)


def prepare_data(
    graph: graphs.ModelGraph,
    dataset: pd.DataFrame,
    output_dir: pathlib.Path,
    param_cutoff: float,
    duplicate_tolerances: dict = None,  # None means 'use defaults'
) -> pd.DataFrame:
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

    # Everything is validation except for what we set as training
    dataset[keys.TRAINING_LABEL] = keys.VALIDATE

    # This will set the training label to DUPLICATE for all but one entry in each cluster of identical inputs
    dataset = graph.identify_duplicates(dataset, duplicate_tolerances)

    # Train only on the first iteration
    for path, sc_rows in datasets.iter_self_consistent_paths(dataset):
        for pair, rows in datasets.iter_atom_idx_pairs(sc_rows):
            min_iter = rows[keys.UV_ITER].min()
            first_iter = rows[rows[keys.UV_ITER] == min_iter]
            assert len(first_iter) == 1  # nosec

            # Train from first iteration
            dataset.loc[first_iter.index, keys.TRAINING_LABEL] = keys.TRAIN

    # Make a plot showing the baseline parity results
    plot_baseline(dataset, output_dir)

    return dataset


def plot_baseline(dataset: pd.DataFrame, output_dir: pathlib.Path):
    dataset = dataset.copy()

    # Train only on the first iteration
    for path, sc_rows in datasets.iter_self_consistent_paths(dataset):
        for pair, rows in datasets.iter_atom_idx_pairs(sc_rows):
            min_iter = rows[keys.UV_ITER].min()
            first_iter = rows[rows[keys.UV_ITER] == min_iter]
            assert len(first_iter) == 1  # nosec

            # Train from first iteration
            dataset.loc[sc_rows.index, keys.PARAM_OUT_PREDICTED] = first_iter.iloc[0][
                keys.PARAM_OUT
            ]

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
