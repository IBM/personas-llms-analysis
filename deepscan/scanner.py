"""Subset scanning based on FGSS"""

import time
import numpy as np
from deepscan.util.scoring_functions import ScoringFunctions
from deepscan.scanningops import ScanningOps


def measure_time(func):
    """ """

    def timed(*args, **kw):
        timeststart = time.time() * 1000
        result = func(*args, **kw)
        timeend = time.time() * 1000
        print("%r %6.2f msec" % (func.__name__, timeend - timeststart))

        return result

    return timed


class Scanner:
    """A simplified, faster, exact method but only useable when scoring an individual input"""

    @staticmethod
    # @measure_time
    def fgss_individ_for_nets(
        pvalues,
        a_max=0.5,
        score_function=ScoringFunctions.get_score_bj_fast,
        a_fixed=-1.0,
    ):
        if score_function == "bj":
            score_function = ScoringFunctions.get_score_bj_fast
        if score_function == "hc":
            score_function = ScoringFunctions.get_score_hc_fast
        if score_function == "ks":
            score_function = ScoringFunctions.get_score_ks_fast

        """ This method recognizes that for an individual input, the priority function for a
        fixed alpha threshold results in all nodes having either priority 1 or 0.
        That is, the pmax is either below the threshold.
        or not.  Due to convexity of the scoring function we know elements with tied priority are either all included
        or all excluded.  Therefore, each alpha threshold uniquely defines a single subset of nodes to be scored.
        These are the nodes that have pmax less than threshold.
        This means the individual-input scanner is equivalent to sorting pmax values and iteratively adding the
        next largest pmax.  There are at most O(N) of these subsets to consider.  Sorting requires O(N ln N).
        There is no iterative ascent required and no special choice of alpha thresholds for speed improvements."""
        pmaxes = np.reshape(pvalues[:, 0], pvalues.shape[0])
        # pmaxes = np.reshape(pvalues[:,1], pvalues.shape[0]) #should be number of columns/nodes
        # we can ignore any pmax that is greater than a_max; this makes sorting faster
        # all the pvalues less than equal a_max are kept by nonzero result of the comparison

        # this code does a single score of all and only pvalues less than a fixed alpha (i.e. 0.05 or 0.2).
        # it is a naive alternative to the sorting and scoring done by LTSS
        if a_fixed > 0.0:
            potential_thresholds = pmaxes[
                np.flatnonzero(pmaxes <= a_fixed)
            ]  # only grab less than fixed
            size_fixed_alpha = potential_thresholds.shape[0]  # how many under fixed
            score_fixed_alpha = score_function(
                np.array([size_fixed_alpha]),
                np.array([size_fixed_alpha]),
                np.array([a_fixed]),
            )  # only score once

            node_sub_fixed_alpha = np.flatnonzero(
                pvalues[:, 1] <= a_fixed
            )  # the nodes are all under fixed
            image_sub_fixed_alpha = np.array(
                [0]
            )  # a single entry with index 0 -- single mode only
            return (
                score_fixed_alpha[0],
                image_sub_fixed_alpha,
                node_sub_fixed_alpha,
                a_fixed,
            )

        potential_thresholds = pmaxes[np.flatnonzero(pmaxes <= a_max)]

        if len(potential_thresholds) == 0:
            print("Warning!  No pmaxes less than alpha max  Setting Score to 0.0")
            best_score = 0.0
            optimal_alpha = 0.0
            node_sub = np.array([0])
            image_sub = np.array([0])
            return (best_score, image_sub, node_sub, optimal_alpha)

        # sorrted_unique provides our alpha thresholds that we will scan
        # count_unique (in cumulative format) will provide the number of observations less than correspoding alpha
        sorted_unique, count_unique = np.unique(
            potential_thresholds, return_counts=True
        )
        # print(sorted_unique)
        cumulative_count = np.cumsum(count_unique)

        # In individual input case we have n_a = N, so cumulative count is used for both.
        # sorted_unique provides the increasing alpha values that need to be checked.
        vector_of_scores = score_function(
            cumulative_count, cumulative_count, sorted_unique
        )

        # scoring completed, now grab the max (and index)
        best_score_idx = np.argmax(vector_of_scores)
        best_score = vector_of_scores[best_score_idx]
        optimal_alpha = sorted_unique[best_score_idx]
        # best_size = cumulative_count[best_score_idx]

        # In order to determine which nodes are included, we look for all pmaxes less than or equal best alpha
        node_sub = np.flatnonzero(pvalues[:, 1] <= optimal_alpha)
        # in the individual input case there's only 1 possible subset of inputs to return - a 1x1 with index 0
        image_sub = np.array([0])

        return (best_score, image_sub, node_sub, optimal_alpha)

    """ Subset scanning based on FGSS """

    @staticmethod
    # @measure_time
    def fgss_for_nets(
        pvalues,
        pred_classes,
        a_max=0.5,
        restarts=10,
        image_to_node_init=False,
        constraint=None,
        score_function=ScoringFunctions.get_score_bj_fast,
    ):
        """iterates between images and nodes, each time performing NPSS efficient maximization"""

        if score_function == "bj":
            score_function = ScoringFunctions.get_score_bj_fast
        if score_function == "hc":
            score_function = ScoringFunctions.get_score_hc_fast
        if score_function == "ks":
            score_function = ScoringFunctions.get_score_ks_fast

        best_score = -100000

        if len(pvalues) < restarts:
            restarts = len(pvalues)

        for r_indx in range(
            0, restarts
        ):  # do random restarts to come close to global maximum
            if r_indx < 2:
                if r_indx == 0:
                    # all 1's for number of rows
                    image_to_node = True
                    indices_of_seeds = np.arange(pvalues.shape[0])

                    if constraint == "class":
                        unique_classes = set(pred_classes)

                        for pclass in unique_classes:
                            mask = [x == pclass for x in pred_classes]
                            indices_of_seeds = np.flatnonzero(mask)

                            (
                                best_score_from_restart,
                                best_image_sub_from_restart,
                                best_node_sub_from_restart,
                                best_alpha_from_restart,
                            ) = ScanningOps.single_restart(
                                pvalues,
                                pred_classes,
                                constraint,
                                a_max,
                                indices_of_seeds,
                                image_to_node,
                                score_function,
                            )
                            if best_score_from_restart > best_score:
                                best_score = best_score_from_restart
                                image_sub = best_image_sub_from_restart
                                node_sub = best_node_sub_from_restart
                                optimal_alpha = best_alpha_from_restart
                    else:
                        (
                            best_score_from_restart,
                            best_image_sub_from_restart,
                            best_node_sub_from_restart,
                            best_alpha_from_restart,
                        ) = ScanningOps.single_restart(
                            pvalues,
                            pred_classes,
                            constraint,
                            a_max,
                            indices_of_seeds,
                            image_to_node,
                            score_function,
                        )

                        if best_score_from_restart > best_score:
                            best_score = best_score_from_restart
                            image_sub = best_image_sub_from_restart
                            node_sub = best_node_sub_from_restart
                            optimal_alpha = best_alpha_from_restart

                elif r_indx == 1:
                    # all 1's for number of cols
                    image_to_node = False
                    indices_of_seeds = np.arange(pvalues.shape[1])

                    (
                        best_score_from_restart,
                        best_image_sub_from_restart,
                        best_node_sub_from_restart,
                        best_alpha_from_restart,
                    ) = ScanningOps.single_restart(
                        pvalues,
                        pred_classes,
                        constraint,
                        a_max,
                        indices_of_seeds,
                        image_to_node,
                        score_function,
                    )

                    if best_score_from_restart > best_score:
                        best_score = best_score_from_restart
                        image_sub = best_image_sub_from_restart
                        node_sub = best_node_sub_from_restart
                        optimal_alpha = best_alpha_from_restart

                # Finished A Restart

            else:
                image_to_node = image_to_node_init
                # For cases where restart is greater than 2 choose random indices
                # some some randomizing and only leave in a random number of rows of pvalues TODO
                prob = np.random.uniform(0, 1)
                if image_to_node:
                    indices_of_seeds = np.random.choice(
                        np.arange(pvalues.shape[0]),
                        int(pvalues.shape[0] * prob),
                        replace=False,
                    )
                else:
                    indices_of_seeds = np.random.choice(
                        np.arange(pvalues.shape[1]),
                        int(pvalues.shape[1] * prob),
                        replace=False,
                    )
                while indices_of_seeds.size == 0:
                    # eventually will make non zero
                    prob = np.random.uniform(0, 1)
                    if image_to_node:
                        indices_of_seeds = np.random.choice(
                            np.arange(pvalues.shape[0]),
                            int(pvalues.shape[0] * prob),
                            replace=False,
                        )
                    else:
                        indices_of_seeds = np.random.choice(
                            np.arange(pvalues.shape[1]),
                            int(pvalues.shape[1] * prob),
                            replace=False,
                        )

                indices_of_seeds.astype(int)
                # process a random subset of rows of pvalues array
                (
                    best_score_from_restart,
                    best_image_sub_from_restart,
                    best_node_sub_from_restart,
                    best_alpha_from_restart,
                ) = ScanningOps.single_restart(
                    pvalues,
                    pred_classes,
                    constraint,
                    a_max,
                    indices_of_seeds,
                    image_to_node,
                    score_function,
                )

                if best_score_from_restart > best_score:
                    best_score = best_score_from_restart
                    image_sub = best_image_sub_from_restart
                    node_sub = best_node_sub_from_restart
                    optimal_alpha = best_alpha_from_restart

                # Finished A Restart
        return (best_score, image_sub, node_sub, optimal_alpha)
