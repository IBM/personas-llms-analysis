"""Utility methods"""

import numpy as np
from deepscan.scanner import Scanner
from deepscan.util.metrics import Metrics as ScanMetrics


def write_hyperparams(args, resultsfile):
    """Helper function to write hyperparamters to the results file"""
    outfile = open(resultsfile, "a+")

    outfile.write(
        "hyperparams: clean_ssize:{} anom_ssize:{} bgddata:{} cleandata:{} anomdata:{} model:{} conditional:{} layers:{} run:{} restarts:{} customfunction: {} constraint: {} scorefunc: {} pvaltest: {}\n".format(
            str(args.clean_ssize),
            str(args.anom_ssize),
            str(args.bgddata),
            str(args.cleandata),
            str(args.anomdata),
            str(args.model),
            str(args.conditional),
            str(",".join(map(str, args.layers))),
            str(args.run),
            str(args.restarts),
            args.customfunction,
            args.constraint,
            args.scorefunc,
            args.pvaltest,
        )
    )
    outfile.close()


def scan(
    pvalranges,
    pred_classes,
    clean_ssize,
    anom_ssize,
    restarts,
    conditional=None,
    constraint=None,
    score_function="bj",
    a_fixed=-1.0,
    sampled_indices=None,
):
    """
    Call the scanning methods
    """
    if clean_ssize + anom_ssize == 1:
        # there's only a single individual input in the test set
        if conditional:
            best_scores = []
            image_subs = []
            node_subs = []
            optimal_alphas = []

            for label in range(pvalranges.shape[1]):
                pval_r = pvalranges[:, label, :].reshape(
                    pvalranges.shape[0], pvalranges.shape[2]
                )

                best_score, image_sub, node_sub, optimal_alpha = (
                    Scanner.fgss_individ_for_nets(
                        pval_r, score_function=score_function, a_fixed=a_fixed
                    )
                )

                best_scores.append(best_score)
                image_subs.append(image_sub)
                node_subs.append(node_sub)
                optimal_alphas.append(optimal_alpha)

            min_indx = best_scores.index(min(best_scores))
            best_score = best_scores[min_indx]
            image_sub = image_subs[min_indx]
            node_sub = node_subs[min_indx]
            optimal_alpha = optimal_alphas[min_indx]

        else:
            best_score, image_sub, node_sub, optimal_alpha = (
                Scanner.fgss_individ_for_nets(
                    pvalranges, score_function=score_function, a_fixed=a_fixed
                )
            )
            return best_score, image_sub, node_sub, optimal_alpha

    else:
        if conditional:
            best_scores = []
            image_subs = []
            node_subs = []
            optimal_alphas = []

            for label in range(pvalranges.shape[2]):
                pval_r = pvalranges[:, :, label, :].reshape(
                    pvalranges.shape[0], pvalranges.shape[1], pvalranges.shape[3]
                )

                best_score, image_sub, node_sub, optimal_alpha = Scanner.fgss_for_nets(
                    pval_r,
                    pred_classes,
                    restarts=restarts,
                    constraint="class",
                    score_function=score_function,
                )
                best_scores.append(best_score)
                image_subs.append(image_sub)
                node_subs.append(node_sub)
                optimal_alphas.append(optimal_alpha)

            min_indx = best_scores.index(min(best_scores))
            best_score = best_scores[min_indx]
            image_sub = image_subs[min_indx]
            node_sub = node_subs[min_indx]
            optimal_alpha = optimal_alphas[min_indx]

        else:
            best_score, image_sub, node_sub, optimal_alpha = Scanner.fgss_for_nets(
                pvalranges,
                pred_classes,
                restarts=restarts,
                constraint=constraint,
                score_function=score_function,
            )
            return best_score, image_sub, node_sub, optimal_alpha

    image_sub_indices = image_sub
    if sampled_indices is not None:
        clean_len = len((sampled_indices[0]))
        sampled_indices = list(sampled_indices[0]) + list(sampled_indices[1])
        image_sub_indices = []
        for image in image_sub:
            image_sub_index = sampled_indices[image]
            if image > clean_len:
                # then image is anom
                image_sub_index = image_sub_index * -1
            image_sub_indices.append(image_sub_index)
        image_sub_indices = np.array(image_sub_indices)

    return best_score, image_sub_indices, node_sub, optimal_alpha


def scan_write_metrics(
    pvalranges,
    pred_classes,
    clean_ssize,
    anom_ssize,
    resultsfile,
    restarts,
    conditional,
    constraint,
    score_function,
    a_fixed,
    sampled_indices,
):
    """
    Call the scanning methods and write required metrics to resultsfile
    :param pvalranges:  pvalue ranges numpy array of
        dimension (clean_ssize + anom_ssize, no of attributes (nodes), 2)
    :param clean_ssize: sample size of clean records
    :param anom_ssize: sample size of anomalous records
    :param resultsfile: output file for the results from scanning
    :param restarts: how many iterative restarts for fgss
    """
    scanmetrics = ScanMetrics(resultsfile)

    best_score, image_sub, node_sub, optimal_alpha = scan(
        pvalranges,
        pred_classes,
        clean_ssize,
        anom_ssize,
        restarts,
        conditional,
        constraint,
        score_function,
        a_fixed,
        sampled_indices,
    )

    scanmetrics.get_metrics_from_run(
        best_score, image_sub, node_sub, optimal_alpha, clean_ssize, anom_ssize
    )


def customsort(act, conditional):
    """sorting method to support datastructre for conditional activations"""
    sortedact = {}

    if conditional:
        for label in act:
            sortedact[label] = np.sort(act[label], axis=0)
        return sortedact
    else:
        return np.sort(act, axis=0)
