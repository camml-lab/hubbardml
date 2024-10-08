import functools
import logging
import pathlib
from typing import Tuple, Union

import hydra
from hydra.core import hydra_config
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.utils.data
import torch.optim.lr_scheduler

import hubbardml.utils
from hubbardml import engines
from hubbardml import graphs
from hubbardml import keys
from hubbardml import models
from hubbardml import training

import utils

_LOGGER = logging.getLogger(__name__)

TRAINER = "trainer.pth"
MODEL = "model.pth"

# Batch size used whenever we are just doing inference (e.g. validation/test), this should be as large as
# possible while still fitting in memory
INFERENCE_BATCH_SIZE = 2048

AUTO_BATCH_SIZE = "auto"
AUTO_BATCH_LIMIT = 2048


def checkpoint(trainer: training.Trainer, output_dir: pathlib.Path):
    _LOGGER.info(trainer.status())
    torch.save(trainer, output_dir / TRAINER)  # nosec B614
    torch.save(trainer.model, output_dir / MODEL)  # nosec B614


def init_data(cfg: omegaconf.DictConfig) -> hubbardml.GraphData:
    """Create the model graph and prepare the dataset for the training experiment"""
    # Create the graph that will be used to handle data preparation for the neural network
    graph = hydra.utils.instantiate(cfg["graph"])

    # Prepare the data
    return hubbardml.GraphData(graph, cfg["dataset"])


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def train(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
    _LOGGER.info("Configuration (%s):\n%s", output_dir, omegaconf.OmegaConf.to_yaml(cfg))

    if "seed" in cfg:
        hubbardml.utils.random_seed(cfg["seed"])
    if "dtype" in cfg:
        dtype = torch.from_numpy(np.zeros(0, np.dtype(cfg["dtype"]))).dtype
        torch.set_default_dtype(dtype)

    # Data initialisation
    graph_data = init_data(cfg)

    # Turn the experiment data into an actual training dataset, with test/valid split
    dataset = hydra.utils.instantiate(cfg["prepare_data"])(graph_data)

    # Any initial analysis of the data before training
    utils.analyse_dataset(graph_data, output_dir)
    if cfg.get("analyse_data") is not None:
        hydra.utils.instantiate(cfg["analyse_data"])(dataset, output_dir)

    # Training
    do_train(dataset, graph_data.graph, cfg, output_dir)


def do_train(
    dataset: pd.DataFrame,
    graph: graphs.ModelGraph,
    cfg: omegaconf.DictConfig,
    output_dir: pathlib.Path,
) -> training.Trainer:
    _LOGGER.info(
        "Data splits set:\n%s",
        dataset[[keys.TRAINING_LABEL, keys.LABEL]]
        .groupby(keys.TRAINING_LABEL)
        .value_counts(sort=False),
    )

    device = cfg.get("device")
    target_column = cfg["target_column"]
    train_data, validate_data = get_hubbard_datasets(
        dataset, graph, device, target_column=target_column
    )

    # Create the model
    model_kwargs = {}
    if cfg["rescale"] is not None:
        method = cfg["rescale"]
        model_kwargs["rescaler"] = models.Rescaler.from_data(dataset[target_column], method=method)

    model = hydra.utils.instantiate(cfg["model"])(graph, **model_kwargs)
    _LOGGER.info("Created model:\n%s", model)

    # Create optimiser, trainer, etc
    optimiser = hydra.utils.instantiate(cfg["optimiser"])(model.parameters())
    # Try a decaying learning schedule

    batch_size = cfg["train"]["batch_size"]
    if batch_size == AUTO_BATCH_SIZE:
        batch_size = int(max((len(train_data) / AUTO_BATCH_LIMIT) ** 2 * AUTO_BATCH_LIMIT, 1))
        _LOGGER.info("Using auto batch size: %i/%i", batch_size, len(train_data))

    trainer: training.Trainer = hydra.utils.instantiate(cfg["trainer"])(
        model=model,
        opt=optimiser,
        train_data=torch.utils.data.DataLoader(train_data, batch_size=batch_size),
        validate_data=torch.utils.data.DataLoader(validate_data, batch_size=INFERENCE_BATCH_SIZE),
    )
    # Use GPU if asked to
    if "device" in cfg:
        trainer.to(cfg["device"])

    if cfg.get("scheduler") is not None:
        scheduler = hydra.utils.instantiate(cfg["scheduler"])(optimiser)
        trainer.add_trainer_listener(SchedulerListener(scheduler))

    train = cfg["train"]
    outcome = trainer.train(
        min_epochs=train["min_epochs"],
        max_epochs=train["max_epochs"],
        callback=functools.partial(checkpoint, output_dir=output_dir),
        callback_period=50,
    )
    _LOGGER.info("Training finished: %s", outcome)

    # Take the model with the lowest validation loss
    model = trainer.best_model

    # Make predictions using the trained model
    dataset = infer(
        model, dataset, device=device, target_column=target_column, batch_size=INFERENCE_BATCH_SIZE
    )

    dataset.to_json(output_dir / "dataset.json")
    dataset.to_excel(output_dir / "dataset.ods", engine="odf")

    # Perform the analysis and create plots
    utils.analyse_results(dataset, output_dir, trainer=trainer)

    return trainer


def get_hubbard_datasets(
    dataset: pd.DataFrame,
    graph: graphs.ModelGraph,
    device: str = None,
    target_column: str = keys.PARAM_OUT,
) -> Tuple[graphs.HubbardDataset, graphs.HubbardDataset]:
    # Get views on our train and validate data
    df_train = dataset[dataset[keys.TRAINING_LABEL] == keys.TRAIN]
    df_validate = dataset[dataset[keys.TRAINING_LABEL] == keys.VALIDATE]

    train_data = graphs.HubbardDataset(
        graph,
        df_train,
        target_column=target_column,
        dtype=torch.get_default_dtype(),
        device=device,
    )

    validate_data = graphs.HubbardDataset(
        graph,
        df_validate,
        target_column=target_column,
        dtype=torch.get_default_dtype(),
        device=device,
    )

    return train_data, validate_data


def infer(
    model,
    frame: pd.DataFrame,
    device: str = None,
    target_column: str = keys.PARAM_OUT,
    batch_size: int = None,
) -> pd.DataFrame:
    dataset = graphs.HubbardDataset(
        model.graph,
        frame,
        target_column=target_column,
        dtype=torch.get_default_dtype(),
        device=device,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    frame[keys.PARAM_OUT_PREDICTED] = (
        engines.evaluate(model, loader).detach().cpu().numpy().reshape(-1)
    )
    frame[keys.PARAM_DELTA] = frame.apply(
        lambda row: row[keys.PARAM_OUT] - row[keys.PARAM_OUT_PREDICTED], axis=1
    )
    frame[keys.PARAM_OUT_RELATIVE_ERROR] = frame.apply(
        lambda row: abs(row[keys.PARAM_DELTA] / row[keys.PARAM_OUT]), axis=1
    )

    return frame


class SchedulerListener(training.TrainerListener):
    def __init__(self, scheduler: torch.optim.lr_scheduler.LRScheduler):
        self._scheduler = scheduler

    def epoch_ended(self, trainer: training.Trainer, epoch_num: int):
        self._scheduler.step()
        if self._scheduler.get_lr() != self._scheduler.get_last_lr():
            _LOGGER.info("Learning rate changed to: %s", self._scheduler.get_last_lr())


if __name__ == "__main__":
    train()
