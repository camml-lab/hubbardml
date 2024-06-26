import collections.abc
import itertools
import math
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import datasets
from . import keys
from . import utils

plot_colours = [
    "#131313",
    "#BF016B",
    "#CBCCC6",
    "#58A4B0",
]

colourmap = matplotlib.colors.ListedColormap(plot_colours, name="pinkblack")

parameter_colours = dict(zip((keys.PARAM_U, keys.PARAM_V), plot_colours))
hp_ml_colours = dict(zip((keys.PARAM_OUT, keys.PARAM_OUT_PREDICTED), plot_colours))
train_validate_colours = {
    keys.TRAIN: plot_colours[2],
    keys.VALIDATE: plot_colours[1],
    keys.TEST: plot_colours[0],
    keys.REFERENCE: plot_colours[3],
}

element_colours = {
    "C": "#909090",
    "Ni": "#50D050",
    "O": "#FF0D0D",
    "Co": "#F090A0",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Ti": "#BFC2C7",
}

DEFAULT_ALPHA = 0.6


def _minmax(df, *keys):
    minval = min(df[key].min() for key in keys)
    maxval = max(df[key].max() for key in keys)

    return minval, maxval


def plot_series(ax, xdat, ydat, color, label, size=50):
    ax.plot(
        xdat,
        ydat,
        c=color,
        marker="o",
        alpha=1,
        linestyle="dashed",
        markerfacecolor="white",
        zorder=1,
    )
    ax.scatter(
        xdat,
        ydat,
        label=label,
        c=color,
        marker=matplotlib.markers.MarkerStyle("o", fillstyle="none"),
        s=size,
        zorder=2,
    )


def create_parity_plot(
    df: pd.DataFrame,
    axis_label: str = None,
    title: str = None,
) -> plt.Figure:
    """
    Create parity plot from validation and training label

    :param df: the dataframe
    :param axis_label: xy axis label
    :param title: plot title
    :return: the matplotlib figure object
    """
    validate = df[df[keys.TRAINING_LABEL] == keys.VALIDATE]
    train = df[df[keys.TRAINING_LABEL] == keys.TRAIN]

    fig = plt.figure(figsize=(3.5, 3.5))
    fig.suptitle(title)

    minval, maxval = _minmax(df, keys.PARAM_OUT, keys.PARAM_OUT_PREDICTED)

    ranges = (minval - 0.1 * np.abs(minval), maxval + 0.1 * np.abs(maxval))

    plt.xlim(ranges)
    plt.ylim(ranges)
    plt.plot(ranges, ranges, c="black", zorder=3)
    if axis_label:
        plt.xlabel(f"{axis_label} training")
        plt.ylabel(f"{axis_label} prediction")

    plt.scatter(
        train[keys.PARAM_OUT],
        train[keys.PARAM_OUT_PREDICTED],
        label=f"Training {utils.to_mev_string(datasets.rmse(train, keys.TRAIN))}",
        s=5,
        c=train[keys.COLOUR],
        alpha=DEFAULT_ALPHA,
        zorder=2,
    )
    plt.scatter(
        validate[keys.PARAM_OUT],
        validate[keys.PARAM_OUT_PREDICTED],
        label=f"Validation {utils.to_mev_string(datasets.rmse(validate, keys.VALIDATE))}",
        s=42,
        alpha=0.5,
        c=validate[keys.COLOUR],
        zorder=2,
    )
    plt.legend()

    return fig


def create_progression_plots(df, yrange: float = None, num_cols=3, max_plots=None, scale=1.0):
    paths = datasets.get_self_consistent_paths(df)
    total = 0
    for path in paths:
        sc_frame = df[df[keys.DIR].str.startswith(path)]
        pairs = sc_frame.apply(lambda row: (row[keys.ATOM_1_IDX], row[keys.ATOM_2_IDX]), axis=1)
        total += len(pairs.unique())

    if max_plots is not None:
        total = min(total, max_plots)
    num_rows = math.ceil(total / num_cols)

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        sharex=True,
        figsize=(6 * scale * num_cols, 4 * scale * num_rows),
    )

    # Set axes labels on outermost subplots
    for row in range(num_rows):
        ax = axs[row][0]
        ax.set_ylabel(f"Hubbard ${df.iloc[0][keys.PARAM_TYPE]}$ (eV)")

    for col in range(num_cols):
        ax = axs[-1][col]
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Iteration")

    axs = list(itertools.chain(*axs))

    axis_no = 0
    for path in paths:
        sc_frame = df[df[keys.DIR].str.startswith(path)]

        # Iterate over atom index pairs
        for pair, pair_rows in datasets.iter_atom_idx_pairs(sc_frame):
            sorted_df = pair_rows.sort_values(by=[keys.UV_ITER])
            iters = sorted_df[keys.UV_ITER]
            row = sorted_df.iloc[0]

            # Set up the axes
            ax = axs[axis_no]
            ax.set_title(
                "".join(map(str, [row[keys.ATOM_1_IN_NAME], row[keys.ATOM_2_IN_NAME]])),
                y=0.8,
            )

            if yrange is not None:
                param_values = sorted_df[keys.PARAM_OUT].tolist()
                if keys.PARAM_OUT_PREDICTED in sorted_df:
                    param_values.extend(sorted_df[keys.PARAM_OUT_PREDICTED].tolist())

                midpoint = np.mean(param_values)
                ax.set_ylim((midpoint - 0.5 * yrange, midpoint + 0.5 * yrange))

            plot_series(
                ax,
                iters.tolist(),
                sorted_df[keys.PARAM_OUT].tolist(),
                color=hp_ml_colours[keys.PARAM_OUT],
                label="HP",
            )

            if keys.PARAM_OUT_PREDICTED in sorted_df:
                plot_series(
                    ax,
                    iters,
                    sorted_df[keys.PARAM_OUT_PREDICTED],
                    color=hp_ml_colours[keys.PARAM_OUT_PREDICTED],
                    label="ML",
                )

            ax.legend()
            axis_no += 1

            if max_plots is not None and axis_no >= max_plots:
                return fig

    return fig


def split_plot(
    df: pd.DataFrame,
    category_key: str,
    axis_label: str = None,
    title: str = None,
    prediction_key: str = keys.PARAM_OUT_PREDICTED,
    show_overall_rmse=True,
) -> plt.Figure:
    """Create a parity plot with separate series based on the category key"""
    categories = df[category_key].unique()
    fig = plt.figure(figsize=(3.5, 3.5))
    fig.suptitle(title)

    # Calculate the plot ranges
    minval, maxval = _minmax(df, keys.PARAM_OUT, prediction_key)
    ranges = (minval - 0.1 * np.abs(minval), maxval + 0.1 * np.abs(maxval))

    plt.plot(ranges, ranges, c="black", zorder=3)
    for category in categories:
        # For the validate set extract the rows for this specie
        subset = df[df[category_key] == category]

        vals = subset[keys.PARAM_OUT]
        predicted = subset[prediction_key]

        rmse = np.linalg.norm(vals - predicted) / np.sqrt(len(vals))
        plt.scatter(
            vals,
            predicted,
            label=f"{category} {utils.to_mev_string(rmse)}",
            c=subset[keys.COLOUR],
            alpha=DEFAULT_ALPHA,
        )

    if axis_label:
        plt.xlabel(f"{axis_label} target")
        plt.ylabel(f"{axis_label} prediction")

    plt.xlim(ranges)
    plt.ylim(ranges)

    if show_overall_rmse:
        handles, _labels = fig.gca().get_legend_handles_labels()
        overall_rmse = datasets.rmse(df, prediction_key=prediction_key, training_label=None)
        handles.append(
            matplotlib.patches.Patch(
                color="none", label=f"Overall {utils.to_mev_string(overall_rmse)}"
            )
        )
        plt.legend(handles=handles)
    else:
        plt.legend()

    return fig


def plot_training_curves(training_run: pd.DataFrame, logscale=True) -> plt.Figure:
    fig = plt.figure(figsize=(7, 3))
    fig.suptitle("Training curves")
    ax = fig.gca()

    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    plt.cla()
    # plt.scatter(
    #     training_run[keys.ITER],
    #     training_run[keys.TRAIN_LOSS],
    #     s=5,
    #     label="Train",
    #     c=train_validate_colours[keys.TRAIN],
    #     alpha=0.8,
    # )
    plot_series(
        ax,
        training_run[keys.EPOCH],
        training_run[keys.TRAIN_LOSS],
        label="Training",
        color=train_validate_colours[keys.TRAIN],
        size=5,
        # alpha=0.8,
    )
    if keys.VALIDATE_LOSS in training_run:
        # plt.scatter(
        #     training_run[keys.ITER],
        #     training_run[keys.VALIDATE_LOSS],
        #     s=5,
        #     label="Validate",
        #     c=train_validate_colours[keys.VALIDATE],
        #     alpha=0.8,
        # )
        plot_series(
            ax,
            training_run[keys.EPOCH],
            training_run[keys.VALIDATE_LOSS],
            label="Validation",
            color=train_validate_colours[keys.VALIDATE],
            size=5,
        )
    plt.legend()
    ax = plt.gca()
    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    return fig


def plot_param_histogram(
    df: pd.DataFrame,
    x_label: str = "Hubbard param. (eV)",
    y_label: str = None,
    title: str = None,
    param_col: str = keys.PARAM_OUT,
    bins=20,
    labels: Dict[str, str] = None,
    density=True,
    group_by=keys.SPECIES,
    hist_type="stepfilled",
    stacked=False,
) -> plt.Figure:
    """Plot a histogram of the Hubbard parameter values, split by element"""
    y_label = y_label or "Frequency" if density else "Count"

    labels = labels or {}
    fig, ax = plt.subplots()
    fig.suptitle(title)

    minval = df[param_col].min()
    maxval = df[param_col].max()
    binwidth = (maxval - minval) / bins
    bins_ = np.arange(minval, maxval + binwidth, binwidth)

    kwargs = dict(
        histtype=hist_type,
        alpha=0.8,
        density=density,
        ec="k",
        stacked=stacked,
        # log=True,
    )
    values = []
    labels = []
    colours = []
    for species, frame in df.groupby(group_by):
        vals, bins = np.histogram(frame[param_col], bins=bins_)
        values.append(vals)
        if isinstance(species, str):
            group_label = species
        else:
            group_label = "-".join(species)
        group_label += f" (avg {frame[param_col].mean():.3f})"
        labels.append(group_label)
        colours.append(frame.iloc[0][keys.COLOUR])

    ax.hist(
        np.expand_dims(bins[:-1], axis=1).repeat(len(labels), axis=1),
        bins,
        weights=np.vstack(values).T,
        color=colours,
        label=labels,
        **kwargs,
    )

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    ax.legend()

    return fig
