from collections import Counter
import numpy as np


def get_anom_nodes(fn="output/"):
    """
    Extract from deepscan output file the list of nodes that were found abnormal across all runs.
    Depending the configuration of the scan this file contains other information such as:
    optimal alphas, subset of statements, subset of nodes, metrics per run such as precision
    and recall.

    Args:
    - fn: path to deepscan output file.
    """

    nodes_samples = []
    stmt_samples = []
    precision_samples = []
    recall_samples = []
    optimal_alpha_samples = []

    with open(fn, "r") as f:
        for line in f.readlines()[:-1]:
            if "inf" in line.rstrip().split(" "):
                pass
            else:
                (
                    _,
                    precision,
                    recall,
                    _,
                    _,
                    optimal_alpha,
                    anom_node,
                    anom_stmts,
                ) = line.rstrip().split(" ")
                nodes = anom_node.strip().split(",")
                stmts = anom_stmts.strip().split(",")
                if nodes != [""]:
                    nodes = list(map(int, nodes))
                stmts = list(map(int, stmts))
                nodes_samples.append(np.array(nodes))
                stmt_samples.append(stmts)
                if precision != "None":
                    precision_samples.append(float(precision))
                    recall_samples.append(float(recall))
                optimal_alpha_samples.append(float(optimal_alpha))

    return (
        nodes_samples,
        stmt_samples,
        precision_samples,
        recall_samples,
        optimal_alpha_samples,
    )


def most_common_nodes(output_files_dict, vector_size=4096):
    """
    This function extract the most common nodes detected across all deepscan runs.

    Args:
    - output_files_dict: dictionary with dimension name and output file from deepscan run
    - vector_size: we use the size of the vector to include as much nodes as possible.
    If all nodes of the vector were present on all runs, we want to include them.
    This value can be changed for example to the avg cardinality value of returned subsets.
    """

    topics_nodes = {}

    for name, filename in output_files_dict.items():
        runs, _, _, _, _ = get_anom_nodes(filename)
        flat_nodes = [node for run in runs for node in run]
        flat_nodes_count = Counter(flat_nodes)
        list_nodes_common, counts = zip(*flat_nodes_count.most_common(vector_size))
        topics_nodes[name] = set(list_nodes_common)

    return topics_nodes
