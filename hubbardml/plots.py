import itertools
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import keys

plot_colours = [
    "#222222",
    "#BF016B",
    "#999999",
    # '#FFFFFF',
]

colourmap = matplotlib.colors.ListedColormap(plot_colours, name="pinkblack")

parameter_colours = dict(zip((keys.PARAM_U, keys.PARAM_V), plot_colours))
hp_ml_colours = dict(zip((keys.PARAM_OUT, keys.PARAM_OUT_PREDICTED), plot_colours))

element_colours = {
    "C": "#909090",
    "Ni": "#50D050",
    "O": "#FF0D0D",
    "Co": "#F090A0",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Ti": "#BFC2C7",
}


def _minmax(df, *keys):
    minval = min(df[key].min() for key in keys)
    maxval = max(df[key].max() for key in keys)

    return minval, maxval


def create_parity_plot(df, axis_label=None, title=None):
    test = df[df[keys.TRAINING_LABEL] == keys.TEST]
    train = df[df[keys.TRAINING_LABEL] == keys.TRAIN]

    fig = plt.figure(figsize=(4, 4))
    fig.suptitle(title)

    minval, maxval = _minmax(df, keys.PARAM_OUT, keys.PARAM_OUT_PREDICTED)

    ranges = (minval - 0.1 * np.abs(minval), maxval + 0.1 * np.abs(maxval))

    plt.xlim(ranges)
    plt.ylim(ranges)
    plt.plot(ranges, ranges, c=plot_colours[1])
    if axis_label:
        plt.xlabel(f"{axis_label} training")
        plt.ylabel(f"{axis_label} prediction")

    # plt.scatter

    plt.scatter(
        train[keys.PARAM_OUT],
        train[keys.PARAM_OUT_PREDICTED],
        label="Training",
        s=5,
        c=train[keys.COLOUR],
    )
    plt.scatter(
        test[keys.PARAM_OUT],
        test[keys.PARAM_OUT_PREDICTED],
        label="Test",
        s=42,
        alpha=0.5,
        c=test[keys.COLOUR],
    )
    plt.legend()

    return fig


def create_progression_plots(df, root_path, yrange: float = None, num_cols=3):
    sc_frame = df[df[keys.DIR].str.startswith(root_path)]

    # Get atom index pairs
    pairs = sc_frame.apply(lambda row: (row[keys.ATOM_1_IDX], row[keys.ATOM_2_IDX]), axis=1)
    unique_pairs = pairs.unique()

    num_rows = math.ceil(len(unique_pairs) / 3)

    fig = plt.figure(figsize=(6 * num_cols, 4 * num_rows))
    gs = fig.add_gridspec(num_rows, num_cols, hspace=0)
    axs = gs.subplots(
        sharex="col",
    )
    fig.suptitle(sc_frame.iloc[0][keys.DIR], y=0.9)

    axs = list(itertools.chain(*axs))
    for pair, ax in zip(unique_pairs, axs):
        pair_rows = sc_frame[sc_frame.index.isin(pairs[pairs == pair].index)]
        sorted_df = pair_rows.sort_values(by=[keys.UV_ITER])
        iters = sorted_df[keys.UV_ITER]
        row = sorted_df.iloc[0]

        # Set up the axes
        ax.set_title(
            "".join(map(str, [row[keys.ATOM_1_IN_NAME], row[keys.ATOM_1_IN_NAME]])),
            y=0.85,
        )

        if yrange is not None:
            param_values = sorted_df[keys.PARAM_OUT].tolist()
            if keys.PARAM_OUT_PREDICTED in sorted_df:
                param_values.extend(sorted_df[keys.PARAM_OUT_PREDICTED].tolist())

            midpoint = np.mean(param_values)
            ax.set_ylim((midpoint - 0.5 * yrange, midpoint + 0.5 * yrange))

        ax.plot(
            iters.tolist(),
            sorted_df[keys.PARAM_OUT].tolist(),
            label="HP",
            c=hp_ml_colours[keys.PARAM_OUT],
            marker="o",
            linestyle="dashed",
        )
        if keys.PARAM_OUT_PREDICTED in sorted_df:
            ax.plot(
                iters,
                sorted_df[keys.PARAM_OUT_PREDICTED],
                label="ML",
                c=hp_ml_colours[keys.PARAM_OUT_PREDICTED],
                marker="o",
                linestyle="dashed",
            )

        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Hubbard ${row[keys.PARAM_TYPE]}$ (eV)")

        ax.legend()

    return fig


def split_plot(df, category_key, axis_label=None, title=None):
    categories = df[category_key].unique()
    # test_elements = np.array(df.iloc[test_indices][keys.ATOM_1_ELEMENT])
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle(title)

    # Calculate the plot ranges
    minval, maxval = _minmax(df, keys.PARAM_OUT, keys.PARAM_OUT_PREDICTED)
    ranges = (minval - 0.1 * np.abs(minval), maxval + 0.1 * np.abs(maxval))

    plt.plot(ranges, ranges, c=plot_colours[1])
    for category in categories:
        # For the test set extract the rows for this specie
        subset = df[df[category_key] == category]

        vals = subset[keys.PARAM_OUT]
        predicted = subset[keys.PARAM_OUT_PREDICTED]

        rmse = np.linalg.norm(vals - predicted) / np.sqrt(len(vals))
        plt.scatter(vals, predicted, label=f"{category} {rmse:.3f} eV", c=subset[keys.COLOUR])

    if axis_label:
        plt.xlabel(f"{axis_label} target")
        plt.ylabel(f"{axis_label} prediction")

    plt.xlim(ranges)
    plt.ylim(ranges)

    plt.legend()

    return fig


def plot_training_curves(training_run: pd.DataFrame, logscale=True):
    fig = plt.figure(figsize=(6, 3))
    fig.suptitle("Training curves")

    plt.ylabel("Loss")
    plt.xlabel("Iteration")

    plt.cla()
    plt.scatter(
        training_run[keys.ITER],
        training_run[keys.TRAIN_LOSS],
        s=5,
        label="Train",
        c=plot_colours[0],
        alpha=0.8,
    )
    if keys.TEST_LOSS in training_run:
        plt.scatter(
            training_run[keys.ITER],
            training_run[keys.TEST_LOSS],
            s=5,
            label="Test",
            c=plot_colours[1],
            alpha=0.8,
        )
    plt.legend()
    ax = plt.gca()
    if logscale:
        ax.set_yscale("log")

    return fig
