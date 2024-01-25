import functools
import itertools
import json
import logging
import os
import pathlib
import operator
from typing import Iterable, Tuple, List

import hydra
from hydra.core import hydra_config
import omegaconf
import pandas as pd

import hubbardml
from hubbardml import datasets
from hubbardml import keys
from hubbardml import graphs
from hubbardml import similarities
from hubbardml import training

import run

_LOGGER = logging.getLogger(__name__)

RESULTS_FILE = "hubbard_u_olivines.json"
MATERIALS = "LiFePO4", "LiMnPO4", "LiFe0.5Mn0.5PO4"
OCCUPATIONS = "0.00", "0.25", "0.50", "0.75", "1.00"


class Keys:
    MATERIAL = "material"
    NUM_OCCUPATIONS = "num_occupations"
    TRAIN_OCCUPATIONS = "occupation"
    MODEL_RMSE = "model_rmse"
    TRAIN_RMSE = "train_rmse"
    REF_RMSE = "ref_rmse"

    OCCUPATION = "occupation"


def prepare_occupations_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Filter to get out the materials that we want"""
    # Extract just the materials we're interested in
    mask = functools.reduce(
        operator.or_, [dataset[keys.DIR].str.contains(material) for material in MATERIALS]
    )
    dataset = dataset[mask]

    # Add a column that contains the lithium concentration
    for occ in OCCUPATIONS:
        dataset.loc[dataset[keys.DIR].str.contains(occ), Keys.OCCUPATION] = float(occ)

    return dataset


def get_results_frame(output_dir) -> pd.DataFrame:
    if os.path.exists(output_dir / RESULTS_FILE):
        with open(RESULTS_FILE, "r") as file:
            return pd.DataFrame(json.load(file))
    else:
        return pd.DataFrame(
            [],
            columns=(
                Keys.MATERIAL,
                Keys.NUM_OCCUPATIONS,
                Keys.TRAIN_OCCUPATIONS,
                Keys.MODEL_RMSE,
                Keys.REF_RMSE,
                Keys.TRAIN_RMSE,
            ),
        )


def init_data(cfg: omegaconf.DictConfig) -> hubbardml.GraphData:
    # Create the data handler that we'll be using to handle inputs
    graph: graphs.ModelGraph = hydra.utils.instantiate(cfg["graph"])

    # Prepare the data
    return hubbardml.GraphData(graph, cfg["dataset"])


@hydra.main(version_base="1.3", config_path=".", config_name="olivines")
def train_olivines(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
    _LOGGER.info("Configuration (%s):\n%s", output_dir, omegaconf.OmegaConf.to_yaml(cfg))

    graph_data = init_data(cfg)
    data = graph_data.dataset
    graph = graph_data.graph

    # Prepare the occupations
    data = prepare_occupations_data(data)

    data.to_json(output_dir / "dataset.json")

    results_frame = get_results_frame(output_dir)
    for material in MATERIALS:
        _LOGGER.info("Starting material: %s", material)
        # Get the data just for this material
        dataset = data[data[keys.DIR].str.contains(material)].copy()
        dataset = graph_data.identify_duplicates(dataset, group_by=[Keys.OCCUPATION])

        trainer = train_reference(dataset, graph, material, cfg, output_dir)
        ref_rmse = trainer.data_logger.as_dataframe().validate_rmse.min()
        _LOGGER.info("Reference RMSE: %d", ref_rmse)

        # Now train the combinations of occupations
        for num_occupations in range(1, len(OCCUPATIONS)):
            for occs in itertools.combinations(OCCUPATIONS, num_occupations):
                # Label this combination of parameters
                label = f"{material}_{len(occs)}_{'-'.join(occs)}"

                trainer = train_occupations(
                    dataset,
                    graph,
                    label,
                    occs,
                    cfg,
                    output_dir,
                )

                # Calculate the minimum RMSE reached during training
                progress = trainer.data_logger.as_dataframe()
                min_rmse = progress.validate_loss.min() ** 0.5
                train_rmse = progress.train_loss.min() ** 0.5

                _LOGGER.info("Min: %f, Ref: %f, Train: %f", min_rmse, ref_rmse, train_rmse)

                # Append a row to the results and save
                results_frame.loc[len(results_frame)] = [
                    material,
                    len(occs),
                    occs,
                    min_rmse,
                    ref_rmse,
                    train_rmse,
                ]
                results_frame.to_json(output_dir / RESULTS_FILE)


def create_training_frame(dataset: pd.DataFrame, occs: Iterable[str]) -> pd.DataFrame:
    # Set everything to duplicate and then weed out the ones we want in train/validate
    dataset[keys.TRAINING_LABEL] = keys.DUPLICATE

    # Set the training set to those that correspond to the allowed occupations
    for occ in occs:
        dataset.loc[dataset[keys.DIR].str.contains(occ), keys.TRAINING_LABEL] = keys.TRAIN

    # From those that we want to validate one, choose one per cluster
    for cluster_id in dataset.loc[
        dataset[keys.TRAINING_LABEL] == keys.DUPLICATE, similarities.CLUSTER_ID
    ].unique():
        dataset.loc[
            dataset[dataset[similarities.CLUSTER_ID] == cluster_id].first_valid_index(),
            keys.TRAINING_LABEL,
        ] = keys.VALIDATE

    return dataset


def train_reference(
    dataset: pd.DataFrame,
    graph: graphs.ModelGraph,
    label: str,
    cfg: omegaconf.DictConfig,
    output_dir: pathlib.Path,
) -> training.Trainer:
    dataset = datasets.split_by_cluster(
        dataset,
        category=[Keys.OCCUPATION],
        ignore_already_labelled=True,
    )

    ref_dir = output_dir / label
    ref_dir.mkdir(exist_ok=True)
    return run.do_train(dataset, graph, cfg, ref_dir)


def train_occupations(
    dataset: pd.DataFrame,
    graph: graphs.ModelGraph,
    label: str,
    occs: List,
    cfg: omegaconf.DictConfig,
    output_dir: pathlib.Path,
) -> training.Trainer:
    dataset = create_training_frame(dataset.copy(), occs)

    _LOGGER.info("%s training split:\n%s", label, dataset[keys.TRAINING_LABEL].value_counts())

    train_dir = output_dir / label
    train_dir.mkdir()
    return run.do_train(dataset, graph, cfg, train_dir)


if __name__ == "__main__":
    train_olivines()
