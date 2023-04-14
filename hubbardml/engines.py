import abc
import collections
import math
from typing import Callable, Any, Union, Iterable, Generator
import uuid

import torch
import torch.utils.data

__all__ = ("evaluate",)


class EngineListener:
    """Listen for events emitted by the engine"""

    def epoch_starting(self, engine: "Engine"):
        """Epoch is starting"""

    def batch_starting(self, engine: "Engine", batch_idx: int, x, y):
        """A batch is being run through the model"""

    def batch_ended(self, engine: "Engine", batch_idx, x, y, y_pred):
        """Batch finished running through model"""

    def epoch_ended(self, engine: "Engine", num_batches: int):
        """The epoch ended"""


class Metric(EngineListener):
    def __init__(self, save_as: str):
        self.save_as = save_as

    def epoch_starting(self, engine: "Engine"):
        self.reset()

    def batch_ended(self, engine: "Engine", batch_idx, x, y, y_pred):
        with torch.no_grad():
            self.update(y, y_pred)

    def epoch_ended(self, engine: "Engine", batch_size: int):
        engine.metrics[self.save_as] = self.compute()

    @abc.abstractmethod
    def reset(self):
        """Reset the metric accumulators"""

    @abc.abstractmethod
    def update(self, y, y_pred):
        """Update accumulators with this iteration of data"""

    @abc.abstractmethod
    def compute(self) -> Union[float, torch.Tensor]:
        """Compute the metric"""


class Mae(Metric):
    def __init__(self, save_as: str):
        super().__init__(save_as)
        self._sum_of_absolute_errors = 0.0
        self._num_examples = 0

    def reset(self):
        self._sum_of_absolute_errors = 0.0
        self._num_examples = 0

    def update(self, y, y_pred):
        y_pred, y = y_pred.detach(), y.detach()
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        self._sum_of_absolute_errors += torch.sum(absolute_errors)
        self._num_examples += y.shape[0]

    def compute(self):
        return self._sum_of_absolute_errors.item() / self._num_examples


class Rmse(Metric):
    def __init__(self, save_as: str):
        super().__init__(save_as)
        self._mae = Mae(save_as)

    def reset(self):
        self._mae.reset()

    @abc.abstractmethod
    def update(self, y, y_pred):
        self._mae.update(y, y_pred)

    @abc.abstractmethod
    def compute(self) -> Union[float, torch.Tensor]:
        return math.sqrt(self._mae.compute())


class EventGenerator:
    def __init__(self):
        self._event_listeners = {}

    def add_listener(self, listener) -> Any:
        handle = uuid.uuid4()
        self._event_listeners[handle] = listener
        return handle

    def remove_listener(self, handle) -> Any:
        return self._event_listeners.pop(handle)

    def fire_event(self, event_fn: Callable, *args, **kwargs):
        for listener in self._event_listeners.values():
            getattr(listener, event_fn.__name__)(*args, **kwargs)


class Engine:
    """Training engine"""

    def __init__(self, model: torch.nn.Module):
        self._model = model
        self._events = EventGenerator()
        self.metrics = {}

    def add_engine_listener(self, listener: EngineListener):
        return self._events.add_listener(listener)

    def remove_engine_listener(self, handle):
        return self._events.remove_listener(handle)

    def step(self, data: Iterable) -> Generator:
        """Perform one epoch"""
        self._events.fire_event(EngineListener.epoch_starting, self)
        device = self._model.device

        total_batch_size = 0
        for batch_idx, batch in enumerate(data):
            x, y = _to(batch[0], device=device), _to(batch[1], device=device)
            self._events.fire_event(EngineListener.batch_starting, self, batch_idx, x, y)

            y_pred = self._model(x)
            yield batch_idx, x, y, y_pred

            self._events.fire_event(EngineListener.batch_ended, self, batch_idx, x, y, y_pred)
            total_batch_size += len(x)

        self._events.fire_event(EngineListener.epoch_ended, self, total_batch_size)

    def run(self, data: Iterable):
        collections.deque(self.step(data), maxlen=0)


class OutputSaver(EngineListener):
    def __init__(self):
        super().__init__()
        self._outputs = []

    def get_outputs(self) -> torch.Tensor:
        return torch.stack(self._outputs)

    def epoch_starting(self, engine: "Engine"):
        # Reset out outputs
        self._outputs = []

    def batch_ended(self, engine: "Engine", batch_idx, x, y, y_pred):
        self._outputs.append(y_pred)


def evaluate(model: torch.nn.Module, data: Iterable):
    """Evaluate the dataset and return all the model outputs"""
    model.eval()
    engine = Engine(model)
    saver = OutputSaver()
    engine.add_engine_listener(saver)
    with torch.no_grad():
        engine.run(data)

    return saver.get_outputs()


def _to(obj, device):
    """Send a tensor or dictionary of tensors to a particular device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _to(value, device)

        return obj
    else:
        raise TypeError(obj)
