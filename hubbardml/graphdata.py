import hashlib
import logging
import pathlib
from typing import Iterable, Union, Iterator, Tuple, Any

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

    def get_similarity_frames(
        self, group_by=None
    ) -> Iterator[Tuple[Any, pd.DataFrame, pd.DataFrame]]:
        if group_by is None:
            group_by = self._graph.DEFAULT_GROUP_BY

        if group_by is not None:
            iterator = self.dataset.groupby(group_by)
        else:
            iterator = ((None, self._dataset),)

        input_filename = pathlib.Path(self._data_path.name)
        for identifier, group in iterator:
            # Create the filename to use for the cached similarities frame
            parts = [
                input_filename.stem,
                self.graph.__class__.__name__,
                hashlib.shake_256(str(identifier).encode()).hexdigest(10),
            ]
            cache_filename = "_".join(parts) + ".arrow"
            _LOGGER.info("Getting similarities for %s, %s", input_filename, str(identifier))

            if pathlib.Path(cache_filename).exists():
                # Load the cached version
                _LOGGER.info("Reading cached similarities %s", cache_filename)
                similarities_frame = pd.read_feather(cache_filename)
            else:
                # Create the cached version
                _LOGGER.info("Creating similarities frame")
                similarities_frame = self._graph.get_similarity_frame(group, group_by=tuple())
                _LOGGER.info("Saving cached similarities frame to %s", cache_filename)
                similarities_frame.to_feather(cache_filename)

            yield identifier, group, similarities_frame

    def identify_duplicates(
        self,
        dataset: pd.DataFrame,
        group_by: Iterable[str] = tuple(),
        tolerances=None,
    ):
        """Utility function for identifying duplicate entries in the data.  This uses a file to cache the similarities
        frame as this can be expensive to calculate.

        WARNING: Make sure to delete the cache if changes are made to the dataset or model graph
        """
        kwargs = {}
        if tolerances is not None:
            kwargs["tolerances"] = tolerances

        for identifier, group, similarity_frame in self.get_similarity_frames(group_by):
            _LOGGER.info("Identifying duplicates in %s", identifier)
            dups = self._graph.identify_duplicates_(
                dataset.loc[group.index], similarity_frame, **kwargs
            )

            # Update the dataset with results from dups
            dataset.loc[dups.index, dups.columns] = dups

        return dataset
