"""Scanning operations"""

import time
import numpy as np


def measure_time(func):
    """ """

    def timed(*args, **kw):
        timeststart = time.time() * 1000
        result = func(*args, **kw)
        timeend = time.time() * 1000
        print("%r %6.2f msec" % (func.__name__, timeend - timeststart))

        return result

    return timed


class ScanningOps:
    """Specific operations done during scanning"""

    @staticmethod
    def create_alpha_thresholds(pvalues, a_max, thresholds_mode):
        """
        Create alpha tresholds given a specific mode
        """

        if thresholds_mode == "all":
            """
            Here we allow all threshold values up to and including a_max.
            This is the number necessary to guarantee optimality (linearly many)
            but perhaps can be improved speedwise with minimal loss of accuracy
            """
            alpha_thresholds = np.unique(pvalues[:, :, 1])
            # where does a_max fall in check
            last_alpha_index = np.searchsorted(alpha_thresholds, a_max)
            alpha_thresholds = alpha_thresholds[0:last_alpha_index]
            alpha_thresholds = np.append(alpha_thresholds, a_max)

        elif thresholds_mode == "grid":
            start = a_max / 50
            alpha_thresholds = np.linspace(start, a_max, 50)

        elif thresholds_mode == "set_number":
            """
            Here we use a set number of equally spaced thresholds created from np.unique sort.
            Default is 50 thresholds (or close to it)
            """

            set_number = 50

            alpha_thresholds, _ = np.unique(pvalues[:, :, 1], return_counts=True)
            # where does a_max fall in check
            last_alpha_index = np.searchsorted(alpha_thresholds, a_max)
            alpha_thresholds = alpha_thresholds[0:last_alpha_index]

            if len(alpha_thresholds) > set_number:
                step = len(alpha_thresholds) / set_number
                grab_these_indices = range(0, len(alpha_thresholds), step)
                alpha_thresholds = np.take(alpha_thresholds, grab_these_indices)

            alpha_thresholds = np.append(alpha_thresholds, a_max)

        elif thresholds_mode == "single_alpha":
            """"
            This represents a naive alternative to LTSS where a single alpha threshold is used.
            Should be much faster but have poorer detection power as it doesn't allow the data
            to recommend the alpha that maximizes the scoring functions
            """

        else:
            print("Please provide threshold_mode")

        return alpha_thresholds

    @staticmethod
    # @measure_time
    def optimize_in_single_dimension(pvalues, a_max, image_to_node, score_function):
        """
        takes in a subset of pvalues and direction
        image_to_node is BOOL which informs direction."""

        alpha_thresholds = ScanningOps.create_alpha_thresholds(
            pvalues, a_max, thresholds_mode="grid"
        )

        unsort_priority = None

        if image_to_node:
            number_of_elements = pvalues.shape[1]  # searching over j columns
            size_of_given = pvalues.shape[0]  # for fixed this many images
            unsort_priority = np.zeros(
                (pvalues.shape[1], alpha_thresholds.shape[0])
            )  # number of columns
        else:
            number_of_elements = pvalues.shape[0]  # searching over i rows
            size_of_given = pvalues.shape[1]  # for this many fixed nodes
            unsort_priority = np.zeros(
                (pvalues.shape[0], alpha_thresholds.shape[0])
            )  # number of rows

        for elem_indx in range(0, number_of_elements):
            # sort all the range maxes
            if image_to_node:
                # collect ranges over images(rows)
                arg_sort_max = np.argsort(pvalues[:, elem_indx, 1])
                # arg_sort_min = np.argsort(pvalues[:,e,0]) #collect ranges over images(rows)
                completely_included = np.searchsorted(
                    pvalues[:, elem_indx, 1][arg_sort_max],
                    alpha_thresholds,
                    side="right",
                )
            else:
                # collect ranges over nodes(columns)
                arg_sort_max = np.argsort(pvalues[elem_indx, :, 1])
                # arg_sort_min = np.argsort(pvalues[elem_indx,:,0])

                completely_included = np.searchsorted(
                    pvalues[elem_indx, :, 1][arg_sort_max],
                    alpha_thresholds,
                    side="right",
                )

            # print('complete included shape', completely_included.shape)
            # should be num elements by num thresh
            unsort_priority[elem_indx, :] = completely_included

        # print("unsort priority", unsort_priority)
        # want to sort for a fixed thresh (across?)
        arg_sort_priority = np.argsort(-unsort_priority, axis=0)
        # print("arg_sort_priority", arg_sort_priority)

        best_score_so_far = -10000
        best_alpha = -2

        alpha_count = 0
        for alpha_threshold in alpha_thresholds:
            # score each threshold by itself, cumulating priority,
            # cumulating count, alpha stays same.
            alpha_v = np.ones(number_of_elements) * alpha_threshold

            # may need to reverse this?
            n_alpha_v = np.cumsum(
                unsort_priority[:, alpha_count][arg_sort_priority][:, alpha_count]
            )
            count_increments_this = np.ones(number_of_elements) * size_of_given
            n_v = np.cumsum(count_increments_this)

            vector_of_scores = score_function(n_alpha_v, n_v, alpha_v)

            best_score_for_this_alpha_idx = np.argmax(vector_of_scores)
            best_score_for_this_alpha = vector_of_scores[best_score_for_this_alpha_idx]

            if best_score_for_this_alpha > best_score_so_far:
                best_score_so_far = best_score_for_this_alpha
                best_size = best_score_for_this_alpha_idx + 1  # not sure 1 is needed?
                best_alpha = alpha_threshold
                best_alpha_count = alpha_count
            alpha_count = alpha_count + 1

        # after the alpha for loop we now have best score, best alpha, size of best subset,
        # and alpha counter use these with the priority argsort to reconstruct the best subset

        unsort = arg_sort_priority[:, best_alpha_count]

        subset = np.zeros(best_size).astype(int)
        for loc in range(0, best_size):
            subset[loc] = unsort[loc]

        return (best_score_so_far, subset, best_alpha)

    @staticmethod
    # @measure_time
    def single_restart(
        pvalues,
        pred_classes,
        constraint,
        a_max,
        indices_of_seeds,
        image_to_node,
        score_function,
    ):
        """
        Here we control the iteration between images->nodes and nodes->images
        we start with a fixed subset of nodes by default
        We collapse pvalues array to only use those"""

        best_score_so_far = -100000

        count = 0

        while True:
            # these can be moved outside the while loop as only executed first time through??
            if (
                count == 0
            ):  # first time through, we need something initialized depending on direction.
                if image_to_node:
                    sub_of_images = indices_of_seeds
                else:
                    sub_of_nodes = indices_of_seeds

            if image_to_node:  # passed pvalues are only those belonging to fixed images, update nodes in return
                score_from_optimization, sub_of_nodes, optimal_alpha = (
                    ScanningOps.optimize_in_single_dimension(
                        pvalues[sub_of_images, :, :],
                        a_max,
                        image_to_node,
                        score_function,
                    )
                )  # only sending sub of images
            else:  # passed pvalues are only those belonging to fixed nodes, update images in return
                if constraint == "class":
                    unique_classes = set(pred_classes)

                    for pclass in unique_classes:
                        mask = [x == pclass for x in pred_classes]
                        mask = np.array(mask)

                        selected_pvalues = pvalues[mask, :, :]
                        selected_pvalues = selected_pvalues[:, sub_of_nodes, :]

                        score_from_optimization, sub_of_images, optimal_alpha = (
                            ScanningOps.optimize_in_single_dimension(
                                selected_pvalues, a_max, image_to_node, score_function
                            )
                        )  # only sending sub of nodes

                        if (
                            score_from_optimization > best_score_so_far
                        ):  # havent converged yet
                            # update
                            best_score_so_far = score_from_optimization
                            best_sub_of_nodes = sub_of_nodes
                            best_sub_of_images = sub_of_images
                            best_alpha = optimal_alpha

                else:
                    score_from_optimization, sub_of_images, optimal_alpha = (
                        ScanningOps.optimize_in_single_dimension(
                            pvalues[:, sub_of_nodes, :],
                            a_max,
                            image_to_node,
                            score_function,
                        )
                    )  # only sending sub of nodes

            if score_from_optimization > best_score_so_far:  # havent converged yet
                # update
                best_score_so_far = score_from_optimization
                best_sub_of_nodes = sub_of_nodes
                best_sub_of_images = sub_of_images
                best_alpha = optimal_alpha

                image_to_node = not image_to_node  # switch direction!
                count = count + 1  # for printing and
            else:  # converged!  Don't update from most recent optimiztion, return current best
                return (
                    best_score_so_far,
                    best_sub_of_images,
                    best_sub_of_nodes,
                    best_alpha,
                )

            # end do while
