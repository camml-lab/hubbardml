import logging
import pathlib
from typing import Iterable, Union

import pandas as pd

from . import datasets
from . import graphs

__all__ = ("GraphData",)

_LOGGER = logging.getLogger(__name__)


class GraphData:
    def __init__(self, graph: graphs.ModelGraph, data_path: Union[str, pathlib.Path]):
        self._graph = graph
        self._data_path = pathlib.Path(data_path)
        self._dataset = graph.prepare_dataset(datasets.load(self._data_path))

    @property
    def graph(self) -> graphs.ModelGraph:
        return self._graph

    @property
    def data_path(self) -> pathlib.Path:
        return self._data_path

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset

    def get_similarity_frame(self) -> pd.DataFrame:
        input_filename = pathlib.Path(self._data_path.name)

        # Create the filename to use for the cached similarities frame
        cache_filename = (
            "_".join(
                [
                    input_filename.stem,
                    self.graph.__class__.__name__,
                    "-".join(self._graph.DEFAULT_GROUP_BY),
                ]
            )
            + ".json"
        )

        if pathlib.Path(cache_filename).exists():
            # Load the cached version
            _LOGGER.info("Reading cached similarities frame from %s", cache_filename)
            similarities_frame = pd.read_json(cache_filename)
        else:
            # Create the cached version
            _LOGGER.info("Creating similarities frame %s", cache_filename)
            similarities_frame = self._graph.get_similarity_frame(self._dataset)
            _LOGGER.info("Saving cached similarities frame to %s", cache_filename)
            similarities_frame.to_json(cache_filename)

        return similarities_frame

    def identify_duplicates(
        self,
        dataset: pd.DataFrame,
        group_by: Iterable[str] = tuple(),
        tolerances=None,
        inplace=False,
    ):
        """Utility function for identifying duplicate entries in the data.  This uses a file to cache the similarities
        frame as this can be expensive to calculate.

        WARNING: Make sure to delete the cache if changes are made to the dataset or model graph
        """
        if not inplace:
            dataset = dataset.copy()

        similarity_frame = self.get_similarity_frame()

        kwargs = {}
        if tolerances is not None:
            kwargs["tolerances"] = tolerances

        for name, indexes in dataset.groupby(list(group_by)).groups.items():
            _LOGGER.info("Checking for duplicates in group %s", name)
            dups = self._graph.identify_duplicates_(
                dataset.loc[indexes], similarity_frame, **kwargs
            )

            # Update the dataset with results from dups
            dataset.loc[dups.index, dups.columns] = dups

        return dataset
