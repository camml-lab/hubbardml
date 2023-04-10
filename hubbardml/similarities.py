import networkx as nx
import pandas as pd

DUPLICATE_OF = "duplicate_of"


class SimilarityKeys:
    INDEX_PAIR = "index_pair"
    INPUT_DIST = "input_dist"
    DIST_TRACE = "dist_trace"
    DELTA_PARAM = "delta_param"


def deduplicate(data: pd.DataFrame, similarities: pd.DataFrame, input_threshold) -> pd.DataFrame:
    deduplicated = data.copy()

    g = nx.Graph()
    matched = similarities[similarities[SimilarityKeys.INPUT_DIST] < input_threshold]
    g.add_edges_from(matched[SimilarityKeys.INDEX_PAIR])

    for dups in list(nx.connected_components(g)):
        to_remove = list(dups)[1:]
        deduplicated.drop(index=to_remove, inplace=True)

    return deduplicated
