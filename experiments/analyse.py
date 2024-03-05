import logging
import pathlib

import hydra
from hydra.core import hydra_config
import numpy as np
import omegaconf
import torch
import torch.utils.data
import torch.optim.lr_scheduler

import hubbardml.utils

import utils

_LOGGER = logging.getLogger(__name__)

TRAINER = "trainer.pth"
MODEL = "model.pth"

AUTO_BATCH_SIZE = "auto"
AUTO_BATCH_LIMIT = 2048
BINS = 5_000


def init_data(cfg: omegaconf.DictConfig) -> hubbardml.GraphData:
    """Create the model graph and prepare the dataset for the training experiment"""
    # Create the graph that will be used to handle data preparation for the neural network
    graph = hydra.utils.instantiate(cfg["graph"])

    # Prepare the data
    return hubbardml.GraphData(graph, cfg["dataset"])


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def analyse(cfg: omegaconf.DictConfig) -> None:
    output_dir = pathlib.Path(hydra_config.HydraConfig.get().runtime.output_dir)
    _LOGGER.info("Configuration (%s):\n%s", output_dir, omegaconf.OmegaConf.to_yaml(cfg))

    if "dtype" in cfg:
        dtype = torch.from_numpy(np.zeros(0, np.dtype(cfg["dtype"]))).dtype
        torch.set_default_dtype(dtype)

    # Data initialisation
    graph_data = init_data(cfg)

    utils.analyse_dataset(graph_data, output_dir)


if __name__ == "__main__":
    analyse()
