import functools
import logging
import operator
import uuid

import networkx as nx
import pandas as pd

from . import keys

_LOGGER = logging.getLogger(__name__)

DUPLICATE_OF = "duplicate_of"
CLUSTER_ID = "cluster_id"


class SimilarityKeys:
    INDEX_PAIR = "index_pair"
    INPUT_DIST = "input_dist"
    DIST_TRACE = "dist_trace"
    DELTA_PARAM_IN = "delta_param"


def get_similarity_groups(similarities: pd.DataFrame, tolerances: dict) -> nx.Graph:
    graph = nx.Graph()
    filter_op = functools.reduce(
        operator.and_, [similarities[name] < value for name, value in tolerances.items()]
    )
    matched = similarities[filter_op]
    graph.add_edges_from(matched[SimilarityKeys.INDEX_PAIR])

    return graph


def identify_duplicates(
    data: pd.DataFrame, similarities: pd.DataFrame, tolerances: dict
) -> pd.DataFrame:
    if tolerances is None or len(tolerances) == 0:
        raise ValueError("Must provide tolerances for similarity")

    graph = get_similarity_groups(similarities, tolerances)
    # Add all the nodes because those that aren't in a cluster won't appear in the graph otherwise
    # and our goal is to treat unique rows as a single-entry cluster
    graph.add_nodes_from(data.index)

    # Remove any nodes from the graph that aren't in the dataset
    diff = set(graph.nodes) - set(data.index)
    graph.remove_nodes_from(diff)

    total_removed = 0
    for dups in list(nx.connected_components(graph)):
        data.loc[list(dups), CLUSTER_ID] = str(uuid.uuid4())
        to_remove = list(dups)[1:]  # Keep only the first one
        data.loc[to_remove, keys.TRAINING_LABEL] = keys.DUPLICATE

        total_removed += len(to_remove)
        # deduplicated.drop(index=to_remove, inplace=True)

    _LOGGER.info(
        "Identified %i duplicates, total unique: %i",
        total_removed,
        len(data[CLUSTER_ID].unique()),
    )

    return data
