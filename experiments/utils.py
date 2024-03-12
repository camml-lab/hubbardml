import logging
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import hubbardml.utils
from hubbardml import keys
from hubbardml import datasets
from hubbardml import plots

_LOGGER = logging.getLogger(__name__)

BINS = 5_000


def analyse_dataset(graph_data: hubbardml.GraphData, output_dir: pathlib.Path) -> None:
    """Perform some analysis on the training data to gather stats and make plots"""

    # Analyse the species in this dataset
    species_counts = graph_data.dataset[keys.SPECIES].value_counts()
    _LOGGER.info("Species counts:\n%s", species_counts)

    ax = species_counts.plot.bar()
    ax.set_ylabel("Number of data points")
    fig = ax.get_figure()
    fig.savefig(output_dir / "species_hist.pdf", bbox_inches="tight")
    plt.close(fig)

    # Analyse similarities
    # _LOGGER.info("Analysing similarities frame")
    # similarities = graph_data.get_similarity_frame()
    # for key in ("occs_sum", "occs_prod"):
    #     ax = similarities[key].hist(bins=BINS, log=False)
    #     ax.axvline(hubbardml.graphs.DEFAULT_OCCS_TOL)
    #     ax.set_xlim(left=0.0)
    #     fig = ax.get_figure()
    #     fig.set_size_inches(16, 2)
    #     fig.savefig(output_dir / f"{key}_similarity.pdf", bbox_inches="tight")
    #     plt.close(fig)

    # Plot histogram of the Hubbard values
    _LOGGER.info("Analysing parameter distributions")

    # Get just the final (self-consistent) values
    last_iter_subframes = []
    for _path, sc_rows in datasets.iter_self_consistent_paths(graph_data.dataset):
        # Get the maximum iteration reached
        max_iter = sc_rows[keys.UV_ITER].max()
        # Get the rows containing the last iteration
        max_iter_rows = sc_rows[sc_rows[keys.UV_ITER] == max_iter]
        last_iter_subframes.append(max_iter_rows)

    # Remove any duplicates
    last_iter_frame = pd.concat(last_iter_subframes)
    # num_inc_duplicates = len(last_iter_frame)
    # last_iter_frame = last_iter_frame.drop_duplicates(hubbardml.similarities.CLUSTER_ID)
    # _LOGGER.info("Removed %n/%n duplicates", len(last_iter_frame) - num_inc_duplicates, num_inc_duplicates)

    # Plot the histogram
    fig = plots.plot_param_histogram(last_iter_frame, bins=20)

    fig.savefig(output_dir / "param_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
