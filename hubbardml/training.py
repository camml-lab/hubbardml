import collections
import contextlib
import copy
import itertools
from typing import List, Callable, Any
import uuid

import e3psi
import mincepy
import matplotlib.patches
import pandas as pd
import torch
import torch.utils.data

from . import keys
from . import engines
from . import graphs
from . import models
from . import plots
from . import utils

__all__ = "TrainingResult", "Trainer", "train_model"

TrainingResult = collections.namedtuple("TrainingResult", "model df trainer")

TRAIN_MAX_EPOCHS = "max_epochs"
TRAIN_OVERFITTING = "overfitting"
TRAIN_STOP = "stop"

DEFAULT_MAX_EPOCHS = 30_000
DEFAULT_OVERFITTING_WINDOW = 400
DEFAULT_BATCH_SIZE = 4096


def train_model(
    param_type: str,
    df: pd.DataFrame,
    label: str,
    species: List[str] = None,
    create_plots=True,
    max_epochs=DEFAULT_MAX_EPOCHS,
) -> TrainingResult:
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    species = species or (
        set(df[keys.ATOM_1_ELEMENT].unique()) | set(df[keys.ATOM_2_ELEMENT].unique())
    )

    # Create the model
    scaler = models.Rescaler.from_data(df[keys.PARAM_OUT])
    if param_type == "V":
        model = models.VModel(graphs.VGraph(species), rescaler=scaler)
    elif param_type == "U":
        model = models.UModel(graphs.UGraph(species), rescaler=scaler)
    else:
        raise ValueError(f"Parameter type must be 'U' or 'V', got {param_type}")

    model.to(dtype=dtype, device=device)

    return _do_train(
        model,
        param_type,
        df,
        create_plots=create_plots,
        label=label,
        min_iters=5000,
        max_epochs=max_epochs,
    )


class TrainerListener:
    def epoch_started(self, trainer: "Trainer", epoch_num: int):
        """Called when an epoch starts"""

    def epoch_ended(self, trainer: "Trainer", epoch_num: int):
        """Called when an epoch ends"""


class DataLogger(TrainerListener):
    def __init__(self):
        super().__init__()
        self._data_log = []

    def epoch_started(self, trainer: "Trainer", epoch_num):
        self._data_log.append({"epoch": epoch_num})

    def log(self, name: str, value):
        self._data_log[-1][name] = value.detach().item()

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._data_log)


class EarlyStopping(TrainerListener):
    def __init__(self, patience: int, metric="loss"):
        self.patience = patience
        self._metric = metric
        self._num_increases = 0
        self._last_value = None

    def epoch_ended(self, trainer: "Trainer", epoch_num):
        new_value = trainer.validation.metrics[self._metric]
        if self._last_value is not None:
            if new_value > self._last_value:
                self._num_increases += 1
            else:
                # Reset
                self._num_increases = 0

        self._last_value = new_value
        if self._num_increases > self.patience:
            trainer.stop(TRAIN_OVERFITTING)


class CallbackListener(TrainerListener):
    def __init__(self, callback_fn: Callable, callback_period=1):
        self._callback_fn = callback_fn
        self._callback_period = callback_period

    def epoch_ended(self, trainer: "Trainer", epoch_num):
        if epoch_num % self._callback_period == 0:
            self._callback_fn(trainer)


class ModelCheckpointer(TrainerListener):
    def __init__(self, score_function: Callable):
        super().__init__()
        self._score_function = score_function
        self._best_model = None
        self._lowest_score = None

    @property
    def best_model(self):
        return self._best_model

    def epoch_ended(self, trainer: "Trainer", epoch_num):
        current_score = self._score_function(trainer)
        if self._lowest_score is None or current_score < self._lowest_score:
            self._lowest_score = current_score
            self._best_model = copy.deepcopy(trainer.model.state_dict())


MSE = "mse"
RMSE = "rmse"


class Trainer(mincepy.SavableObject):
    TYPE_ID = uuid.UUID("97310848-2192-4766-bbb0-34bb7b122182")

    model = mincepy.field("_model", store_as="model", ref=True)

    @classmethod
    def from_frame(
        cls,
        model: e3psi.Model,
        opt: torch.optim.Optimizer,
        loss_fn: Callable,
        frame: pd.DataFrame,
        batch_size=DEFAULT_BATCH_SIZE,
        overfitting_window=DEFAULT_OVERFITTING_WINDOW,
        target_column: str = keys.PARAM_OUT,
    ) -> "Trainer":
        """This helper will construct a Trainer object by using data from a pandas DataFrame.

        The validation/training data will be extracted using the labels found in the keys.Training_LABEL column.
        """
        # Get views on our train and validate data
        df_train = frame[frame[keys.TRAINING_LABEL] == keys.TRAIN]
        df_validate = frame[frame[keys.TRAINING_LABEL] == keys.VALIDATE]

        train_data = graphs.HubbardDataset(
            model.graph,
            df_train,
            target_column=target_column,
            dtype=torch.get_default_dtype(),
            device=model.device,
        )

        validate_data = graphs.HubbardDataset(
            model.graph,
            df_validate,
            target_column=target_column,
            dtype=torch.get_default_dtype(),
            device=model.device,
        )

        return Trainer(
            model,
            opt,
            loss_fn,
            train_data=torch.utils.data.DataLoader(train_data, batch_size=batch_size),
            validate_data=torch.utils.data.DataLoader(validate_data, batch_size=batch_size),
            overfitting_window=overfitting_window,
        )

    def __init__(
        self,
        model: e3psi.Model,
        opt: torch.optim.Optimizer,
        loss_fn: Callable,
        train_data: torch.utils.data.DataLoader,
        validate_data: torch.utils.data.DataLoader,
        overfitting_window=DEFAULT_OVERFITTING_WINDOW,
    ):
        super().__init__()
        self._model = model
        self._opt = opt
        self._loss_fn = loss_fn

        self._train_data = train_data
        self._validate_data = validate_data

        # Training and validation data loaders
        self._train_loader = train_data
        self._validate_loader = validate_data

        self._events = engines.EventGenerator()
        self.overfitting_window = overfitting_window
        self._stopping = False
        self._stop_msg = None

        # Track all the losses
        self._epoch = 0

        self.training = engines.Engine(self._model)
        self.training.add_engine_listener(engines.Mse("mse"))

        self.validation = engines.Engine(self._model)
        self.validation.add_engine_listener(engines.Mse("mse"))
        self.validation.add_engine_listener(engines.Rmse("rmse"))

        # Save checkpoints using the MSE
        self._checkpointer = ModelCheckpointer(score_function=self._checkpoint_score)
        self.add_trainer_listener(self._checkpointer)

        self._data_logger = DataLogger()
        self.add_trainer_listener(self._data_logger)

    @staticmethod
    def _checkpoint_score(trainer):
        return trainer.validation.metrics["mse"]

    @property
    def best_model(self):
        if self._checkpointer.best_model is None:
            return self.model

        model = copy.deepcopy(self.model)
        model.load_state_dict(self._checkpointer.best_model)
        return model

    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        return self._train_data

    @property
    def validate_loader(self) -> torch.utils.data.DataLoader:
        return self._validate_data

    @property
    def epoch(self):
        return self._epoch

    @property
    def data_logger(self) -> DataLogger:
        return self._data_logger

    def status(self) -> str:
        # Deal with case where metrics are None because we haven't started
        train_mse = (
            f"{self.training.metrics.get(MSE):.5f}"
            if self.training.metrics.get(MSE) is not None
            else None
        )
        valid_mse = (
            f"{self.validation.metrics.get(MSE):.5f}"
            if self.validation.metrics.get(MSE) is not None
            else None
        )
        valid_rmse = (
            f"{self.validation.metrics.get(RMSE):.4f}"
            if self.validation.metrics.get(RMSE) is not None
            else None
        )

        return (
            f"epoch: {self.epoch} "
            f"train: mse {train_mse}, "
            f"valid: mse {valid_mse} rmse {valid_rmse}"
        )

    def __str__(self) -> str:
        return self.status()

    def train(
        self,
        min_epochs=None,
        max_epochs=DEFAULT_MAX_EPOCHS,
        callback: Callable = None,
        callback_period=10,
    ) -> str:
        self._stopping = False
        iterator = itertools.count() if max_epochs == -1 else range(max_epochs)

        listeners = []
        if self.overfitting_window:
            listeners.append(EarlyStopping(self.overfitting_window, MSE))
        if callback is not None:
            listeners.append(CallbackListener(callback, callback_period=callback_period))

        with self.listeners_context(listeners):
            for local_epoch in iterator:
                self._events.fire_event(
                    TrainerListener.epoch_started, self, self.epoch
                )  # EPOCH START
                self._model.train()

                # Iterate over batches
                self._opt.zero_grad()
                for batch_idx, x, y, y_pred in self.training.step(self._train_loader):
                    loss = self._loss_fn(y_pred, y)
                    loss.backward()
                    self._opt.step()
                    self._opt.zero_grad()

                # Now do validation run
                self._model.eval()
                with torch.no_grad():
                    self.validation.run(self._validate_loader)

                self._events.fire_event(TrainerListener.epoch_ended, self, self.epoch)  # EPOCH END

                self._data_logger.log(keys.TRAIN_LOSS, self.training.metrics[MSE])
                self._data_logger.log(keys.VALIDATE_LOSS, self.validation.metrics[MSE])
                self._data_logger.log("validate_rmse", self.validation.metrics[RMSE])

                self._epoch += 1

                if (min_epochs is not None and local_epoch > min_epochs) and self._stopping:
                    break

        if self._stopping:
            return self._stop_msg

        return TRAIN_MAX_EPOCHS

    def plot_training_curves(self, logscale=True):
        return plots.plot_training_curves(self._data_logger.as_dataframe(), logscale=logscale)

    def to(self, device):
        self._model.to(device=device)

    def stop(self, msg: str):
        self._stopping = True
        self._stop_msg = msg

    @contextlib.contextmanager
    def listeners_context(self, listeners: List[TrainerListener]):
        handles = list(map(self.add_trainer_listener, listeners))
        try:
            yield
        finally:
            # Remove all the listeners
            list(map(self.remove_trainer_listener, handles))

    def add_trainer_listener(self, listener) -> Any:
        return self._events.add_listener(listener)

    def remove_trainer_listener(self, handle) -> Any:
        return self._events.remove_listener(handle)


def _do_train(
    model: e3psi.Model,
    param_type,
    df: pd.DataFrame,
    create_plots: bool,
    label: str,
    min_iters=None,
    max_epochs=DEFAULT_MAX_EPOCHS,
):
    if param_type not in ("U", "V"):
        raise ValueError(param_type)

    # Create the trainer
    trainer = Trainer.from_frame(
        model=model,
        opt=torch.optim.Adam(model.parameters(), lr=0.001),
        loss_fn=torch.nn.MSELoss(),
        frame=df,
        overfitting_window=DEFAULT_OVERFITTING_WINDOW,
    )

    # Train the model
    trainer.train(callback=print, callback_period=50, min_epochs=min_iters, max_epochs=max_epochs)

    # Set the predicted values in the dataframe
    val_predictions = (
        engines.evaluate(model, trainer.validate_loader).detach().cpu().numpy().reshape(-1)
    )
    train_predictions = (
        engines.evaluate(model, trainer.train_loader).detach().cpu().numpy().reshape(-1)
    )

    # Get the indices of the training and validation data
    train_idx = df[df[keys.TRAINING_LABEL] == keys.TRAIN].index
    validate_idx = df[df[keys.TRAINING_LABEL] == keys.VALIDATE].index

    df.loc[validate_idx, keys.PARAM_OUT_PREDICTED] = val_predictions
    df.loc[train_idx, keys.PARAM_OUT_PREDICTED] = train_predictions

    if create_plots:
        fig = trainer.plot_training_curves()
        fig.savefig(_plotfile_name(label, f"+{param_type}_training"), bbox_inches="tight")

        validate = df.loc[validate_idx]
        validate_rmse = utils.rmse(validate[keys.PARAM_OUT], validate[keys.PARAM_OUT_PREDICTED])
        fig = plots.create_parity_plot(
            df, title=f"RMSE = {validate_rmse:.3f} eV", axis_label=f"${param_type}$ value (eV)"
        )
        fig.savefig(_plotfile_name(label, f"+{param_type}_parity_training"), bbox_inches="tight")

        fig = plots.split_plot(
            df[df[keys.TRAINING_LABEL] == keys.VALIDATE],
            keys.ATOM_1_ELEMENT,
            axis_label=f"${param_type}$ value (eV)",
            title=f"Validate data, RMSE = {validate_rmse:.2f} eV",
        )
        handles, _labels = fig.gca().get_legend_handles_labels()
        handles.append(
            matplotlib.patches.Patch(
                color="none",
            )
        )

        fig.savefig(_plotfile_name(label, f"+{param_type}_parity_species"), bbox_inches="tight")

    return TrainingResult(model, df, trainer)


def _plotfile_name(label: str, plot_type: str):
    return f"plots/{label}_{plot_type}.pdf"


HISTORIAN_TYPES = (Trainer,)
