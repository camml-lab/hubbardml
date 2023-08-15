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

import hubbardml.utils
from hubbardml import datasets
from hubbardml import engines
from hubbardml import graphs
from hubbardml import keys
from hubbardml import models
from hubbardml import plots
from hubbardml import training

_LOGGER = logging.getLogger(__name__)

TRAINER = "trainer.pth"
MODEL = "model.pth"


def checkpoint(trainer: training.Trainer, output_dir: pathlib.Path):
    _LOGGER.info(trainer.status())
    torch.save(trainer, output_dir / TRAINER)
    torch.save(trainer.model, output_dir / MODEL)


def init_data(
    cfg: omegaconf.DictConfig, output_dir: pathlib.Path
) -> Tuple[pd.DataFrame, graphs.ModelGraph]:
    # Create the graph that will be used to handle data preparation for the neural network
    graph = hydra.utils.instantiate(cfg["graph"])

    # Prepare the data
    dataset = hydra.utils.instantiate(cfg["dataset"])
    dataset = graph.prepare_dataset(dataset)
    # Preprocess the data
    if cfg.get("prepare_data") is not None:
        dataset = hydra.utils.instantiate(cfg["prepare_data"])(
            graph=graph,
            dataset=dataset,
            output_dir=output_dir,
        )

    return dataset, graph


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def train(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
    _LOGGER.info("Configuration (%s):\n%s", output_dir, omegaconf.OmegaConf.to_yaml(cfg))

    if "seed" in cfg:
        hubbardml.utils.random_seed(cfg["seed"])
    if "dtype" in cfg:
        dtype = torch.from_numpy(np.zeros(0, np.dtype(cfg["dtype"]))).dtype
        torch.set_default_dtype(dtype)

    dataset, graph = init_data(cfg, output_dir)
    do_train(dataset, graph, cfg, output_dir)


def do_train(
    dataset, graph, cfg: omegaconf.DictConfig, output_dir: pathlib.Path
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
    batch_size = cfg["train"]["batch_size"]
    trainer: training.Trainer = hydra.utils.instantiate(cfg["trainer"])(
        model=model,
        opt=optimiser,
        train_data=torch.utils.data.DataLoader(train_data, batch_size=batch_size),
        validate_data=torch.utils.data.DataLoader(validate_data, batch_size=batch_size),
    )
    # Use GPU if asked to
    if "device" in cfg:
        trainer.to(cfg["device"])

    train = cfg["train"]
    trainer.train(
        min_epochs=train["min_epochs"],
        max_epochs=train["max_epochs"],
        callback=functools.partial(checkpoint, output_dir=output_dir),
        callback_period=50,
    )

    # Take the model with the lowest validation loss
    model = trainer.best_model

    # Make predictions using the trained model
    dataset = infer(
        model, dataset, device=device, target_column=target_column, batch_size=batch_size
    )

    dataset.to_json(output_dir / "dataset.json")
    dataset.to_excel(output_dir / "dataset.ods", engine="odf")

    # Perform the analysis and create plots
    analyse(dataset, output_dir, trainer=trainer)

    return trainer


def get_hubbard_datasets(
    dataset: pd.DataFrame,
    graph: hubbardml.graphs.ModelGraph,
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


def analyse(
    df: pd.DataFrame,
    output_path: Union[pathlib.Path, str] = ".",
    trainer: training.Trainer = None,
):
    if keys.PARAM_OUT_PREDICTED not in df:
        raise RuntimeError("The predicted parameter values must be set before calling analyse()")

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
            title=f"RMSE = {_to_mev_string(validate_rmse)} ({frac:.2f} holdout)",
        )
        parity_fig.savefig(_plot_path("parity"), bbox_inches="tight")

        for training_label in param_frame[keys.TRAINING_LABEL].unique():
            if training_label is None:
                continue

            subset = param_frame[param_frame[keys.TRAINING_LABEL] == training_label]
            fraction = len(subset) / len(param_frame)
            rmse = datasets.rmse(subset, training_label=training_label)

            # VALIDATE BY SPECIES
            parity_species_fig = plots.split_plot(
                subset,
                keys.LABEL,
                axis_label=f"Hubbard ${param_type}$ (eV)",
                title=f"{training_label} data ({fraction * 100:.0f}%), RMSE = {_to_mev_string(rmse)}",
            )
            parity_species_fig.savefig(
                _plot_path(f"parity_{training_label}_species"), bbox_inches="tight"
            )

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

    return frame


def _to_mev_string(energy):
    return f"{energy * 1000:.0f} meV"


if __name__ == "__main__":
    train()
