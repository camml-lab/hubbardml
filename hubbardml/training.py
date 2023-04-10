import collections
import copy
import uuid
from typing import List, Optional, Callable

import mincepy

import e3psi
import numpy as np
import pandas as pd
import torch

from . import keys
from . import models
from . import plots
from . import utils

__all__ = "TrainingResult", "TrainingInfo", "Trainer", "train_model"

TrainingResult = collections.namedtuple("TrainingResult", "model df trainer")

TRAIN_MAX_ITERS = "max_iters"
TRAIN_OVERFITTING = "overfitting"
TRAIN_STOP = "stop"

DEFAULT_MAX_ITERS = 30_000
DEFAULT_OVERFITTING_WINDOW = 400


class TrainingInfo(collections.namedtuple("TrainingInfo", "iter train_loss validate_loss")):
    def __str__(self):
        return " ".join(f"{field}={getattr(self, field)}" for field in self._fields)


def train_model(
    param_type: str,
    df: pd.DataFrame,
    label: str,
    species: List[str] = None,
    create_plots=True,
    max_iters=DEFAULT_MAX_ITERS,
) -> TrainingResult:
    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    species = species or (
        set(df[keys.ATOM_1_ELEMENT].unique()) | set(df[keys.ATOM_2_ELEMENT].unique())
    )

    # Create the model
    scaler = models.Rescaler.from_data(df[keys.PARAM_OUT])
    if param_type == "V":
        model = models.VModel(species, rescaler=scaler)
    elif param_type == "U":
        model = models.UModel(species, rescaler=scaler)
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
        max_iters=max_iters,
    )


def train(
    model: e3psi.Model,
    opt: torch.optim.Optimizer,
    loss_fn,
    input_train,
    output_train,
    input_validate,
    output_validate,
    min_iters=None,
    max_iters=None,
    overfitting_window: Optional[int] = DEFAULT_OVERFITTING_WINDOW,
    callback: Callable[[TrainingInfo], Optional[str]] = None,
    callback_period=10,
) -> str:
    iter_range = utils._count() if max_iters is None else range(max_iters)

    # Track the losses
    validate_loss = []

    increased_last_n = 0

    for iter in iter_range:
        opt.zero_grad()  # zero the gradient buffers

        output = model(input_train)
        loss = loss_fn(output, output_train)
        loss.backward()

        opt.step()  # Does the update

        # Keep track of the validate loss
        with torch.no_grad():
            validate_loss.append(loss_fn(model(input_validate), output_validate).cpu().item())

        # Check if the validate loss increased
        if len(validate_loss) > 1 and validate_loss[-1] > validate_loss[-2]:
            increased_last_n += 1
        else:
            increased_last_n = 0

        if (
            overfitting_window is not None
            and (min_iters is None or len(validate_loss) > min_iters)
            and increased_last_n >= (0.5 * overfitting_window)
            and validate_loss[-1] > loss.cpu().item()
        ):
            return TRAIN_OVERFITTING

        if callback is not None and (iter % callback_period == 0):
            res = callback(TrainingInfo(iter, loss.cpu().item(), validate_loss[-1]))
            if res == TRAIN_STOP:
                return res

    # Reached the maximum number of iterations
    return TRAIN_MAX_ITERS


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
        overfitting_window=DEFAULT_OVERFITTING_WINDOW,
        target_column: str = keys.PARAM_OUT,
    ) -> "Trainer":
        """This helper will construct a Trainer object by using data from a pandas DataFrame.

        The validation/training data will be extracted using the labels found in the keys.Training_LABEL column.
        """
        # Get views on our train and validate data
        df_train = frame[frame[keys.TRAINING_LABEL] == keys.TRAIN]
        df_validate = frame[frame[keys.TRAINING_LABEL] == keys.VALIDATE]

        dtype = model.dtype
        device = model.device

        input_train = models.create_model_inputs(model.graph, df_train, dtype=dtype, device=device)
        input_validate = models.create_model_inputs(
            model.graph, df_validate, dtype=dtype, device=device
        )

        output_train = torch.tensor(
            df_train[target_column].to_numpy(), dtype=dtype, device=device
        ).reshape(-1, 1)
        output_validate = torch.tensor(
            df_validate[target_column].to_numpy(), dtype=dtype, device=device
        ).reshape(-1, 1)

        return Trainer(
            model,
            opt,
            loss_fn,
            input_train=input_train,
            output_train=output_train,
            input_validate=input_validate,
            output_validate=output_validate,
            overfitting_window=overfitting_window,
        )

    def __init__(
        self,
        model: e3psi.Model,
        opt: torch.optim.Optimizer,
        loss_fn: Callable,
        input_train,
        output_train,
        input_validate,
        output_validate,
        overfitting_window=DEFAULT_OVERFITTING_WINDOW,
    ):
        super().__init__()
        self._model = model
        self._opt = opt
        self._loss_fn = loss_fn

        self._input_train = input_train
        self._output_train = output_train

        self._input_validate = input_validate
        self._output_validate = output_validate

        self.overfitting_window = overfitting_window

        # Track the losses
        self._training_progress = []
        self._best_state = copy.deepcopy(model.state_dict())

    @property
    def best_model(self):
        model = copy.deepcopy(self.model)
        model.load_state_dict(self._best_state)
        return model

    @property
    def progress(self) -> List[TrainingInfo]:
        return self._training_progress

    @property
    def input_train(self):
        return self._input_train

    @property
    def input_validate(self):
        return self._input_validate

    def get_progress_frame(self) -> pd.DataFrame:
        return pd.DataFrame(data=self._training_progress)

    def train(
        self, min_iters=None, max_iters=DEFAULT_MAX_ITERS, callback=None, callback_period=10
    ) -> str:
        start_iter = 0 if not self._training_progress else self._training_progress[-1].iter + 1

        def callback_fn(info: TrainingInfo):
            new_info = info._asdict()
            new_info[keys.ITER] = start_iter + info.iter
            info = TrainingInfo(**new_info)
            self._train_callback(info)

            if callback is not None and info.iter % callback_period == 0:
                return callback(info)

        return train(
            self._model,
            self._opt,
            self._loss_fn,
            self._input_train,
            self._output_train,
            self._input_validate,
            self._output_validate,
            min_iters=min_iters,
            max_iters=max_iters,
            overfitting_window=self.overfitting_window,
            callback=callback_fn,
            callback_period=1,
        )

    def plot_training_curves(self, logscale=True):
        training_run = pd.DataFrame(
            self._training_progress, columns=[keys.ITER, keys.TRAIN_LOSS, keys.VALIDATE_LOSS]
        )
        return plots.plot_training_curves(training_run, logscale=logscale)

    def _train_callback(self, info: TrainingInfo):
        if self._training_progress:
            min_validation_loss = np.min([entry.validate_loss for entry in self._training_progress])
            if info.validate_loss < min_validation_loss:
                # Save the best model state so that we can reuse it later
                self._best_state = copy.deepcopy(self._model.state_dict())

        self._training_progress.append(info)

    def to(self, device):
        self._model.to(device=device)
        self._input_train = _to(self._input_train, device)
        self._output_train = _to(self._output_train, device)
        self._input_validate = _to(self._input_validate, device)
        self._output_validate = _to(self._output_validate, device)


def _to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _to(value, device)

        return obj
    else:
        raise TypeError(obj)


def _do_train(
    model: e3psi.Model,
    param_type,
    df: pd.DataFrame,
    create_plots: bool,
    label: str,
    min_iters=None,
    max_iters=DEFAULT_MAX_ITERS,
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
    trainer.train(callback=print, callback_period=50, min_iters=min_iters, max_iters=max_iters)

    # Set the predicted values in the dataframe
    predicted = model(trainer.input_validate).detach().cpu().numpy().reshape(-1)
    predicted_train = model(trainer.input_train).detach().cpu().numpy().reshape(-1)

    # Get the indices of the training and validation data
    train_idx = df[df[keys.TRAINING_LABEL] == keys.TRAIN].index
    validate_idx = df[df[keys.TRAINING_LABEL] == keys.VALIDATE].index

    df.loc[validate_idx, keys.PARAM_OUT_PREDICTED] = predicted
    df.loc[train_idx, keys.PARAM_OUT_PREDICTED] = predicted_train

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
        fig.savefig(_plotfile_name(label, f"+{param_type}_parity_species"), bbox_inches="tight")

    return TrainingResult(model, df, trainer)


def _plotfile_name(label: str, plot_type: str):
    return f"plots/{label}_{plot_type}.pdf"


HISTORIAN_TYPES = (Trainer,)
