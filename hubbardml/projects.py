import datetime
import pathlib
from typing import Union

import pandas as pd
import torch

from . import engines
from . import graphs
from . import datasets
from . import keys
from . import models
from . import plots
from . import training

__all__ = ("Project",)

PathType = Union[str, pathlib.Path]


class Project:
    DATASET = "dataset.json"
    MODEL = "model.pth"
    TRAINER = "trainer.pth"

    PLOTS = "plots"

    DEFAULT_PARAM_CUTOFF = 0.25  # By default, cut off param_out values lower than this (in eV)

    @classmethod
    def create_path(cls, model_type: str, predict_final=False) -> pathlib.Path:
        """Create a path for a new project"""
        if predict_final:
            name = datetime.datetime.now().strftime(f"{model_type}_final-%Y.%m.%d-%H%M")
        else:
            name = datetime.datetime.now().strftime(f"{model_type}-%Y.%m.%d-%H%M")

        return pathlib.Path(name).absolute()

    @classmethod
    def new(
        cls,
        dataset: Union[str, pathlib.Path, pd.DataFrame],
        model_type: str,
        split_dataset=True,
        training_split=0.2,
        optimiser=torch.optim.Adam,
        optimiser_learning_rate=0.001,
        loss_fn=torch.nn.MSELoss(),
        path: Union[str, pathlib.Path] = None,
        param_cutoff=DEFAULT_PARAM_CUTOFF,
        target_column=keys.PARAM_OUT,
        rescale="mean",
        hidden_layers=None,
    ) -> "Project":
        """

        :param dataset:
        :param model_type: the model to use (U, V, UV, ...)
        :param split_dataset:
        :param training_split:
        :param optimiser:
        :param loss_fn:
        :param path:
        :param param_cutoff: drop all data where the param_out is less than this cutoff value
        :return:
        """
        if path is None:
            path = cls.create_path(model_type)
        else:
            path = pathlib.Path(path).absolute()

        if path.exists():
            raise ValueError(f"Path already exists: {path}")

        # Prepare the dataset
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.copy()
        else:
            # Assume we can load it because we have a path
            dataset = datasets.load(dataset)

        if param_cutoff is not None:
            dataset = dataset[dataset[target_column] > param_cutoff]

        if rescale is not None:
            rescaler = models.Rescaler.from_data(dataset[target_column], method=rescale)
        else:
            rescaler = None

        # Create the classes we need
        graph_class = graphs.GRAPHS[model_type]
        dataset = graph_class.prepare_dataset(dataset)
        species = list(
            pd.concat((dataset[keys.ATOM_1_ELEMENT], dataset[keys.ATOM_2_ELEMENT])).unique()
        )

        # Create the model
        model_class = models.MODELS[model_type]
        models_kwargs = dict(irrep_normalization="component", rescaler=rescaler)
        if hidden_layers is not None:
            models_kwargs["hidden_layers"] = hidden_layers
        model = model_class(graph_class(species), **models_kwargs)

        if split_dataset:
            dataset = datasets.split(
                dataset, method="category", frac=training_split, category=[keys.SPECIES]
            )

        trainer = training.Trainer.from_frame(
            model,
            opt=optimiser(model.parameters(), lr=optimiser_learning_rate),
            loss_fn=loss_fn,
            frame=dataset,
            target_column=target_column,
        )

        # Create the directory and save everything
        path.mkdir(parents=True)
        dataset.to_json(path / cls.DATASET)
        torch.save(model, path / cls.MODEL)
        torch.save(trainer, path / cls.TRAINER)

        return Project(path)

    def __init__(self, path: PathType):
        path = pathlib.Path(path).absolute()
        if not path.is_dir():
            raise ValueError(f"Path either doesn't exist or is not directory: {path}")
        self._path = path
        self._trainer = None
        self._dataset = None

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def plots_path(self) -> pathlib.Path:
        return self.path / self.PLOTS

    def plot_file(self, name: PathType) -> pathlib.Path:
        return self.plots_path / name

    @property
    def model(self):
        return self.trainer.best_model

    @property
    def trainer(self) -> training.Trainer:
        if self._trainer is None:
            self._trainer = torch.load(self.path / self.TRAINER)

        return self._trainer

    @property
    def dataset(self) -> pd.DataFrame:
        if self._dataset is None:
            self._dataset = pd.read_json(self.path / self.DATASET)

        return self._dataset

    @property
    def validate_percentage(self) -> float:
        return float(sum(self.dataset[keys.TRAINING_LABEL] == keys.VALIDATE)) / float(
            len(self.dataset)
        )

    def save(self):
        torch.save(self.trainer, self.path / self.TRAINER)
        torch.save(self.trainer.model, self.path / self.MODEL)

    def train(
        self,
        min_epochs=5_000,
        max_epochs=training.DEFAULT_MAX_EPOCHS,
        print_output=True,
        overfitting_window=None,
    ) -> str:
        callback = print if print_output else None
        if overfitting_window is not None:
            self.trainer.overfitting_window = overfitting_window

        return self.trainer.train(
            min_epochs=min_epochs, max_epochs=max_epochs, callback=callback, callback_period=50
        )

    def infer(self) -> pd.DataFrame:
        val_predictions = (
            engines.evaluate(self.model, self.trainer.validate_loader)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        train_predictions = (
            engines.evaluate(self.model, self.trainer.train_loader)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        df = self.dataset
        # Get the indices of the training and validate data
        train_idx = df[df[keys.TRAINING_LABEL] == keys.TRAIN].index
        validate_idx = df[df[keys.TRAINING_LABEL] == keys.VALIDATE].index

        df.loc[validate_idx, keys.PARAM_OUT_PREDICTED] = val_predictions
        df.loc[train_idx, keys.PARAM_OUT_PREDICTED] = train_predictions

        return df

    def to(self, device):
        self.trainer.to(device)

    def save_training_plots(self, plot_path="plots/", label="", format="pdf"):
        if isinstance(self.model, models.UModel):
            param_type = "U"
        elif isinstance(self.model, models.VModel):
            param_type = "V"
        else:
            raise ValueError(f"Unknown model type: {self.model.__class__.__name__}")

        def _plot_path(plot_type: str) -> pathlib.Path:
            parts = [param_type]
            if label:
                parts.append(label)
            parts.append(plot_type)

            path = pathlib.Path(plot_path) / f"{'_'.join(parts)}.{format}"
            return path

        df = self.infer()
        validate_rmse = datasets.rmse(df)

        # Get the indices of the validation data
        validate_idx = df[df[keys.TRAINING_LABEL] == keys.VALIDATE].index
        # Validation frame
        df_validate = df.loc[validate_idx]

        trainer = self.trainer

        # TRAINING CURVE
        training_fig = trainer.plot_training_curves(logscale=True)
        training_fig.savefig(_plot_path("training_curve"), bbox_inches="tight")

        # TEST/VALIDATE PARITY
        parity_fig = plots.create_parity_plot(
            df,
            axis_label=f"Hubbard ${param_type}$ (eV)",
            title=f"RMSE = {_to_mev_string(validate_rmse)} ({self.validate_percentage:.2f} holdout)",
        )
        parity_fig.savefig(_plot_path("parity_validate"), bbox_inches="tight")

        # VALIDATE BY SPECIES
        parity_species_fig = plots.split_plot(
            df_validate,
            keys.LABEL,
            axis_label=f"Hubbard ${param_type}$ (eV)",
            title=f"Validate data ({self.validate_percentage * 100:.0f}%), RMSE = {_to_mev_string(validate_rmse)}",
        )
        parity_species_fig.savefig(_plot_path("parity_species"), bbox_inches="tight")

        # Iteration progression
        num_cols = 6
        num_rows = 5
        progression_plot = plots.create_progression_plots(
            df[df[keys.PARAM_OUT] > 0.25],
            yrange=0.4,
            num_cols=num_cols,
            max_plots=num_cols * num_rows,
            scale=0.55,
        )
        progression_plot.savefig(_plot_path("convergence"), bbox_inches="tight")


def _to_mev_string(energy):
    return f"{energy * 1000:.0f} meV"
