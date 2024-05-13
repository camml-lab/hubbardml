import logging
import pathlib

import click
import pandas as pd

import utils

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("path", nargs=1, type=pathlib.Path)
def analyse(path: pathlib.Path) -> None:
    results = pd.read_json(path / "dataset.json")
    utils.analyse_results(results, path)


if __name__ == "__main__":
    analyse()
