import logging
import pathlib
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

import hubbardml.utils
from hubbardml import keys
from hubbardml import datasets
from hubbardml import plots
from hubbardml import training

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


def analyse_results(
    df: pd.DataFrame,
    output_path: Union[pathlib.Path, str] = ".",
    trainer: training.Trainer = None,
):
    if keys.PARAM_OUT_PREDICTED not in df:
        raise RuntimeError("The predicted parameter values must be set before calling analyse()")

    _LOGGER.info("Performing analysis and storing to: %s", output_path)

    plots_path = pathlib.Path(output_path) / "plots"
    plots_path.mkdir(exist_ok=True)

    plot_format = "pdf"

    def _plot_path(plot_name: str) -> pathlib.Path:
        path = plots_path / f"{plot_name}.{plot_format}"
        return path

    if trainer is not None:
        # TRAINING CURVE
        training_fig = trainer.plot_training_curves(logscale=True)
        training_fig.savefig(_plot_path("training_curve"), bbox_inches="tight")

    #  PARITY PLOTS
    for param_type in df[keys.PARAM_TYPE].unique():
        param_frame = df[df[keys.PARAM_TYPE] == param_type]
        validate_rmse = datasets.rmse(param_frame, training_label=keys.VALIDATE)
        # Calculate the holdout percentage
        frac = 1.0 - len(param_frame[param_frame[keys.TRAINING_LABEL] == keys.TRAIN]) / len(
            param_frame
        )

        parity_fig = plots.create_parity_plot(
            param_frame,
            axis_label=f"Hubbard ${param_type}$ (eV)",
            title=f"RMSE = {to_mev_string(validate_rmse)} ({frac:.2f} holdout)",
        )
        parity_fig.savefig(_plot_path("parity"), bbox_inches="tight")

        for training_label in param_frame[keys.TRAINING_LABEL].unique():
            if training_label is None:
                continue

            subset = param_frame[param_frame[keys.TRAINING_LABEL] == training_label]

            # VALIDATE BY SPECIES
            parity_species_fig = plots.split_plot(
                subset,
                keys.LABEL,
                axis_label=f"Hubbard ${param_type}$ (eV)",
                title=f"{training_label}".capitalize(),
            )
            parity_species_fig.gca().set_xlabel(f"Hubbard ${param_type}$ from DFPT (eV)")
            parity_species_fig.savefig(
                _plot_path(f"parity_{training_label}_species"), bbox_inches="tight"
            )

            # Create a historgram of the relative errors
            labels = {}
            for species, frame in subset.groupby(keys.LABEL):
                group_label = "-".join(species)
                mean = frame[keys.PARAM_OUT_RELATIVE_ERROR].mean()
                labels[group_label] = f"{group_label} {mean:.3f}"

            plots.plot_param_histogram(
                subset,
                param_col=keys.PARAM_OUT_RELATIVE_ERROR,
                x_label="Relative error",
                labels=labels,
                group_by=keys.LABEL,
                hist_type="bar",
                stacked=True,
                density=False,
            ).savefig(_plot_path(f"relative_error_{training_label}"), bbox_inches="tight")

    # Iteration progression
    num_cols = 6
    num_rows = 5
    progression_plot = plots.create_progression_plots(
        df,
        yrange=0.4,
        num_cols=num_cols,
        max_plots=num_cols * num_rows,
        scale=0.55,
    )
    progression_plot.savefig(_plot_path("convergence"), bbox_inches="tight")


def to_mev_string(energy):
    return f"{energy * 1000:.0f} meV"
