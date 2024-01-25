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
    g = nx.Graph()
    filter_op = functools.reduce(
        operator.and_, [similarities[name] < value for name, value in tolerances.items()]
    )
    matched = similarities[filter_op]
    g.add_edges_from(matched[SimilarityKeys.INDEX_PAIR])

    return g


def identify_duplicates(
    data: pd.DataFrame, similarities: pd.DataFrame, tolerances: dict
) -> pd.DataFrame:
    if tolerances is None or len(tolerances) == 0:
        raise ValueError("Must provide tolerances for similarity")

    deduplicated = data.copy()

    g = get_similarity_groups(similarities, tolerances)
    # Add all the nodes because those that aren't in a cluster won't appear in the graph otherwise
    # and our goal is to treat unique rows as a single-entry cluster
    g.add_nodes_from(data.index)

    # Remove any nodes from the graph that aren't in the dataset
    diff = set(g.nodes) - set(data.index)
    g.remove_nodes_from(diff)

    total_removed = 0
    for dups in list(nx.connected_components(g)):
        deduplicated.loc[list(dups), CLUSTER_ID] = str(uuid.uuid4())
        to_remove = list(dups)[1:]  # Keep only the first one
        deduplicated.loc[to_remove, keys.TRAINING_LABEL] = keys.DUPLICATE

        total_removed += len(to_remove)
        # deduplicated.drop(index=to_remove, inplace=True)

    _LOGGER.info(
        "Identified %i duplicates, total unique: %i",
        total_removed,
        len(deduplicated[CLUSTER_ID].unique()),
    )

    return deduplicated
