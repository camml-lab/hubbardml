import datetime
import pathlib
from typing import Union

import pandas as pd
import torch

from . import datasets
from . import keys
from . import models
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
        predict_final=False,
    ) -> "Project":
        """

        :param dataset:
        :param model_type:
        :param split_dataset:
        :param training_split:
        :param optimiser:
        :param optimiser_learning_rate:
        :param loss_fn:
        :param path:
        :param param_cutoff: drop all data where the param_out is less than this cutoff value
        :return:
        """
        if path is None:
            path = cls.create_path(model_type, predict_final)
        else:
            path = pathlib.Path(path).absolute()

        if path.exists():
            raise ValueError(f"Path already exists: {path}")

        # Prepare the dataset
        if not isinstance(dataset, pd.DataFrame):
            dataset = datasets.load(dataset)

        if predict_final:
            dataset = datasets.generate_final()
            target_param = keys.PARAM_OUT_FINAL
        else:
            target_param = keys.PARAM_OUT

        if param_cutoff is not None:
            dataset = dataset[dataset[keys.PARAM_OUT] > param_cutoff]

        # Create the classes we need
        model_class = models.MODELS[model_type]
        dataset = model_class.prepare_dataset(dataset)
        species = list(pd.concat((dataset[keys.ATOM_1_ELEMENT], dataset[keys.ATOM_2_ELEMENT])).unique())
        model = model_class(species=species)

        if split_dataset:
            dataset = datasets.split(dataset, method="category", frac=training_split, category=[models.SPECIES])

        trainer = training.Trainer.from_frame(
            model,
            opt=optimiser(model.parameters(), lr=optimiser_learning_rate),
            loss_fn=loss_fn,
            frame=dataset,
            target_column=target_param,
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
        return self.trainer.model

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
    def test_percentage(self) -> float:
        return float(sum(self.dataset[keys.TRAINING_LABEL] == keys.TEST)) / float(len(self.dataset))

    def save(self):
        torch.save(self.trainer, self.path / self.TRAINER)
        torch.save(self.trainer.model, self.path / self.MODEL)

    def train(
        self,
        max_iters=10000,
        print_output=True,
        overfitting_window=None,
    ) -> str:
        callback = print if print_output else None
        if overfitting_window is not None:
            self.trainer.overfitting_window = overfitting_window

        return self.trainer.train(max_iters=max_iters, callback=callback, callback_period=50)

    def infer(self) -> pd.DataFrame:
        predicted = self.model(self.trainer.input_test).detach().cpu().numpy().reshape(-1)
        predicted_train = self.model(self.trainer.input_train).detach().cpu().numpy().reshape(-1)

        df = self.dataset
        # Get the indices of the training and test data
        train_idx = df[df[keys.TRAINING_LABEL] == keys.TRAIN].index
        test_idx = df[df[keys.TRAINING_LABEL] == keys.TEST].index

        df.loc[test_idx, keys.PARAM_OUT_PREDICTED] = predicted
        df.loc[train_idx, keys.PARAM_OUT_PREDICTED] = predicted_train

        return df

    def to(self, device):
        self.trainer.to(device)
