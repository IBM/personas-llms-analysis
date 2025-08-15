"""Pvalue ranges calculator - using one tailed test"""

import numpy as np
import time


def measure_time(func):
    """ """

    def timed(*args, **kw):
        timeststart = time.time() * 1000
        result = func(*args, **kw)
        timeend = time.time() * 1000
        print("%r %6.2f msec" % (func.__name__, timeend - timeststart))

        return result

    return timed


class PvalueCalculator:
    """Computes pvalue ranges of a new set of activations given a background set of activations"""

    def __init__(self, sorted_bgd_acts):
        # TODO assert sorted
        self.sorted_bgd_acts = sorted_bgd_acts

    def get_pvalue_ranges_conditional(self, eval_acts):
        """Calculate conditional pvalue ranges"""
        bgd_acts = self.sorted_bgd_acts
        pval_ranges = None

        for label in bgd_acts:
            for eval_label in eval_acts:
                if label == eval_label:
                    if pval_ranges is None:
                        pval_ranges = self.get_pvalue_ranges(
                            eval_acts[eval_label], label
                        )
                    else:
                        _pval_ranges = self.get_pvalue_ranges(
                            eval_acts[eval_label], label
                        )
                        pval_ranges = np.concatenate(
                            (pval_ranges, _pval_ranges), axis=0
                        )

        return pval_ranges

    # @measure_time
    def get_pvalue_ranges_newconditional(self, eval_acts, pvaltest="1tail"):
        """
        Compute the pvalue ranges for a set of activations under evaluation
        :param eval_acts ndarray of dim (datasize, total no of activations)
        :return pvalue ranges ndarray of dim (datasize, attributesize, no_of_clases, 2)
        """
        pval_ranges = None
        bgd_acts = self.sorted_bgd_acts
        for label in bgd_acts:
            if pval_ranges is None:
                pval_ranges = self.get_pvalue_ranges(eval_acts, label, pvaltest)
                pval_ranges = np.expand_dims(pval_ranges, axis=2)

            else:
                pval_ranges_ = self.get_pvalue_ranges(eval_acts, label, pvaltest)
                pval_ranges_ = np.expand_dims(pval_ranges_, axis=2)

                pval_ranges = np.concatenate((pval_ranges, pval_ranges_), axis=2)
        return pval_ranges

    # @measure_time
    def get_pvalue_ranges(self, eval_acts, label=None, pvaltest="1tail"):
        """
        Compute the pvalue ranges for a set of activations under evaluation
        :param eval_acts ndarray of dim (datasize, total no of activations)
        :return pvalue ranges ndarray of dim (datasize, attributesize, 2)
        """
        sorted_bgd_acts = self.sorted_bgd_acts
        if label is not None:
            sorted_bgd_acts = self.sorted_bgd_acts[label]

        bgdrecords_n = sorted_bgd_acts.shape[0]
        evalrecords_n, evalattr_n = eval_acts.shape

        # the 2 is for min and max pair of every evalrecords_nxevalattr_n entry
        out = np.empty((evalrecords_n, evalattr_n, 2))
        for j in range(evalattr_n):
            # left means insert is before ties
            # -> smaller insertion -> larger pvalue after inversion -> pmax

            # import bisect
            # import timeit
            # timeit.timeit(out[:, j, 1] = bisect.bisect_left(sorted_bgd_acts[:, j], eval_acts[:, j]))

            out[:, j, 1] = np.searchsorted(
                sorted_bgd_acts[:, j], eval_acts[:, j], side="left"
            )

            # right means insert is after ties
            # -> larger insertion -> smaller pvalue after inversion -> pmin
            out[:, j, 0] = np.searchsorted(
                sorted_bgd_acts[:, j], eval_acts[:, j], side="right"
            )
        # invert to turn insertion point into #bgd higher  (and ties in case of max)
        out = bgdrecords_n - out

        out[:, :, 0] = np.divide(out[:, :, 0], bgdrecords_n + 1)  # pmins
        out[:, :, 1] = np.divide(out[:, :, 1] + 1, bgdrecords_n + 1)  # pmax

        if pvaltest == "2tail":
            return self.get_two_tailed_pvalue_ranges(out)
        return 1 - out  # 1 - out

    # takes nxmx2 where first tensor is pmin and second is pmax
    # returns nxmx2 but now from 2tailed pvalue
    # TODO! Still very slow with double for loops!!
    def get_two_tailed_pvalue_ranges(self, pvr):
        n = pvr.shape[0]
        m = pvr.shape[1]
        q = np.zeros((n, m, 2))
        for i in range(0, n):  # rows
            for j in range(0, m):  # cols
                if pvr[i, j, 1] < 0.5:  # pmax less than .5
                    q[i, j, 1] = 2.0 * pvr[i, j, 1]
                    q[i, j, 0] = 2.0 * pvr[i, j, 0]
                elif pvr[i, j, 0] > 0.5:  # pmin greater than 0.5
                    q[i, j, 0] = 2.0 * (1.0 - pvr[i, j, 1])  # notice min/max swap
                    q[i, j, 1] = 2.0 * (1.0 - pvr[i, j, 0])  # notice min/max swap
                else:  # straddles   Not quite sure ..
                    q[i, j, 0] = 2.0 * np.minimum((pvr[i, j, 0]), (1.0 - pvr[i, j, 1]))
                    q[i, j, 1] = 1.0
        return q
