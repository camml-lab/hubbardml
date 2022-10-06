from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import keys

plot_colours = [
    "#bf016b",
    "#999999",
    "#222222",
    # '#ffffff',
]

colourmap = matplotlib.colors.ListedColormap(plot_colours, name="pinkblack")

colours = {
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


def create_progression_plots(df, yrange: float = None):
    IN_PAIR = "atom_pair_in_name"
    calculated_colours = {
        "test": "#bf016b",
        "train": "#c28aa9",
    }
    predicted_colours = {"test": "#222222", "train": "#666666"}

    df = df.copy()

    def create_label_pairs(row):
        return " ".join(frozenset([row[keys.ATOM_1_IN_NAME], row[keys.ATOM_2_IN_NAME]]))

    df[IN_PAIR] = df.apply(create_label_pairs, axis=1)

    plots = {}
    for pair_name in df[IN_PAIR].unique():
        fig = plt.figure(figsize=(6, 4))
        plots[pair_name] = fig
        plt.title(str(pair_name))

        # Go to the first step
        step = sorted(list(df[df[IN_PAIR] == pair_name][keys.UV_ITER]))[0]
        row = df[(df[IN_PAIR] == pair_name) & (df[keys.UV_ITER] == step)].iloc[0]

        steps = []
        predicted = []
        calculated = []
        predicted_category = []
        calculated_category = []
        while row is not None:
            steps.append(step)
            predicted.append(row[keys.PARAM_OUT_PREDICTED])
            calculated.append(row[keys.PARAM_OUT])

            cat = row[keys.TRAINING_LABEL]
            predicted_category.append(predicted_colours[cat])
            calculated_category.append(calculated_colours[cat])

            # Move on to next row.
            # 1. Relabel the pair
            # pair_name = frozenset([row[keys.ATOM_1_OUT_NAME], row[keys.ATOM_2_OUT_NAME]])
            # 2. Get the next row
            try:
                df_sorted = df[(df[IN_PAIR] == pair_name) & (df[keys.UV_ITER] > step)]
                row = df_sorted.sort_values(keys.UV_ITER).iloc[0]
            except IndexError:
                row = None
            else:
                step = row[keys.UV_ITER]

        if yrange is not None:
            midpoint = np.vstack((calculated, predicted)).mean()
            plt.ylim((midpoint - 0.5 * yrange, midpoint + 0.5 * yrange))

        plt.plot(steps, calculated, label="HP", c="#bf016b", marker="o", linestyle="dashed")
        plt.plot(steps, predicted, label="ML", c="#222222", marker="o", linestyle="dashed")
        plt.xlabel("Iteration")

        ax = fig.gca()
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        plt.legend()

    return plots


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
