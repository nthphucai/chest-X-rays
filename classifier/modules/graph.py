import pandas as pd 
import numpy as np
import pydot 

def get_label_correlation(df, columns, return_count=False):
    """
    Calculate correlation of columns from data frame
    :param df: pandas dataframe
    :param columns: colunms to calculate correlation
    :param return_count: return occurrence count
    :return: correlation and counts
    """
    counts = pd.DataFrame(columns=columns, index=columns)

    for c1 in columns:
        for c2 in columns:
            counts.loc[c1, c2] = len(df[(df[c1] == 1) & (df[c2] == 1)])

    correlation = counts / np.diag(counts)[:, np.newaxis]

    if return_count:
        return correlation, counts
    else:
        return correlation



def get_adjacency_matrix(smooth_corr, neighbor_ratio=0.2):
    """
    Get adjacency matrix from smoothed correlation
    :param smooth_corr: smoothed correlation matrix as dataframe
    :param neighbor_ratio: how strong neighbor nodes affect main nodes
    :return: adjacency matrix as dataframe
    """
    # over_smoothing problem
    identity = np.identity(smooth_corr.shape[0])
    reweight = smooth_corr - identity
    reweight = reweight * neighbor_ratio / (1 - neighbor_ratio) / (reweight.values.sum(axis=0, keepdims=True) + 1e-8)
    reweight = reweight + identity

    # normalize
    D = reweight.values.sum(axis=1) ** (-0.5)
    D = np.diag(D)
    normalized = D @ reweight.values.transpose() @ D
    return pd.DataFrame(normalized, index=smooth_corr.index, columns=smooth_corr.columns)


def get_graph(corr, threshold=0.4):
    """
    draw a pydot graph of correlation
    :param corr: dataframe of correlation matrix
    :param threshold: threshold to prune correlation
    :return: pydot graph
    """
    smooth_corr = corr >= threshold
    graph = pydot.Dot(graph_type='digraph')

    for c1 in corr.columns:
        node1 = pydot.Node(c1)
        graph.add_node(node1)

        for c2 in corr.columns:
            if c2 != c1:
                node2 = pydot.Node(c2)

                if smooth_corr.loc[c1, c2] != 0:
                    edge = pydot.Edge(node1, node2, label=np.round(corr.loc[c1, c2], decimals=2))
                    graph.add_edge(edge)

    return graph
