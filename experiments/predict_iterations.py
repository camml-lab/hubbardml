import datetime
import json
import logging
import os
import pathlib
from typing import Iterable, Tuple, List

import hydra
from hydra.core import hydra_config
import omegaconf
import pandas as pd

from hubbardml import datasets
from hubbardml import keys
from hubbardml import graphs
from hubbardml import training
from experiment import predict_from_first_iter

import run

_LOGGER = logging.getLogger(__name__)

RESULTS_FILE = "hubbard_u_iterations.json"


class Keys:
    MODEL_RMSE = "model_rmse"
    TRAIN_RMSE = "train_rmse"
    REF_RMSE = "ref_rmse"


def get_results_frame(output_dir) -> pd.DataFrame:
    if os.path.exists(output_dir / RESULTS_FILE):
        with open(RESULTS_FILE, "r") as file:
            return pd.DataFrame(json.load(file))
    else:
        return pd.DataFrame(
            [],
            columns=(
                keys.UV_ITER,
                Keys.MODEL_RMSE,
                Keys.TRAIN_RMSE,
                Keys.REF_RMSE,
            ),
        )


def init_data(cfg: omegaconf.DictConfig) -> Tuple[pd.DataFrame, graphs.ModelGraph]:
    # Create the data handler that we'll be using to handle inputs
    graph: graphs.ModelGraph = hydra.utils.instantiate(cfg["graph"])

    # Prepare the data
    dataset = hydra.utils.instantiate(cfg["dataset"])
    dataset = graph.prepare_dataset(
        dataset
    )  # This sets the self-consistent paths, then we can prepare the occupations

    return dataset, graph


@hydra.main(version_base="1.3", config_path=".", config_name="predict_iterations")
def train_iterations(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
    _LOGGER.info("Configuration (%s):\n%s", output_dir, omegaconf.OmegaConf.to_yaml(cfg))

    data, graph = init_data(cfg)
    data.to_json(output_dir / "dataset.json")

    # Get the iteration numbers that we want to perform experiments on
    uv_iters = sorted(data[keys.UV_ITER].unique())[1:]

    results_frame = get_results_frame(output_dir)
    for uv_iter in uv_iters:
        data = predict_from_first_iter.set_training_labels(data, uv_iter, include_subsequent=False)

        # This will set the training label to DUPLICATE for all but one entry in each cluster of identical inputs
        dups = graph.identify_duplicates(
            data[data[keys.TRAINING_LABEL] == keys.VALIDATE],
            group_by=[keys.SPECIES, keys.SC_PATHS],
        )
        # Copy over the duplicates label to our set
        data.loc[
            dups[dups[keys.TRAINING_LABEL] == keys.DUPLICATE].index, keys.TRAINING_LABEL
        ] = keys.DUPLICATE

        # Now, calculate the reference RMSE on the de-duplicated validation set
        ref_rmse = datasets.rmse(data)
        _LOGGER.info("Reference RMSE: %f", ref_rmse)

        # Create the output directory
        label = f"iter={uv_iter}"
        ref_dir = output_dir / label
        ref_dir.mkdir(exist_ok=True)

        # Do the training
        trainer: training.Trainer = run.do_train(data, graph, cfg, ref_dir)

        # Calculate the minimum RMSE reached during training
        progress = trainer.data_logger.as_dataframe()
        min_rmse = progress.validate_loss.min() ** 0.5
        train_rmse = progress.train_loss.min() ** 0.5

        _LOGGER.info("Min: %f, Train: %f", min_rmse, train_rmse)

        # Append a row to the results and save
        results_frame.loc[len(results_frame)] = [
            uv_iter,
            min_rmse,
            train_rmse,
            ref_rmse,
        ]
        results_frame.to_json(output_dir / RESULTS_FILE)


if __name__ == "__main__":
    train_iterations()
