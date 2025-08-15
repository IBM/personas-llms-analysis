"""Scanner scoring functions"""

import numpy as np


class ScoringFunctions:
    """Scanner scoring functions
    These functions are used in the scanner to determine the score of a subset
    """

    @staticmethod
    def get_score_bj_fast(n_alpha, no_records, alpha):
        """BerkJones
        :param n_alpha: no of records less than alpha
        :param no_records: no of records
        :param n_alpha: alpha threshold
        """

        score = np.zeros(alpha.shape[0])
        inds_tie = n_alpha == no_records
        inds_not_tie = np.logical_not(inds_tie)
        inds_pos = n_alpha > no_records * alpha
        inds_pos_not_tie = np.logical_and(inds_pos, inds_not_tie)

        score[inds_tie] = no_records[inds_tie] * np.log(
            np.true_divide(1, alpha[inds_tie])
        )
        score[inds_pos_not_tie] = n_alpha[inds_pos_not_tie] * np.log(
            np.true_divide(
                n_alpha[inds_pos_not_tie],
                no_records[inds_pos_not_tie] * alpha[inds_pos_not_tie],
            )
        ) + (no_records[inds_pos_not_tie] - n_alpha[inds_pos_not_tie]) * np.log(
            np.true_divide(
                no_records[inds_pos_not_tie] - n_alpha[inds_pos_not_tie],
                no_records[inds_pos_not_tie] * (1 - alpha[inds_pos_not_tie]),
            )
        )

        return score

    @staticmethod
    def get_score_hc_fast(n_alpha, no_records, alpha):
        """HC is similar to a traditional wald test statistic:  (Observed - expected) / standard deviation.
            In this case we use the binomial distribution.  The observed is N_a.  The expected (under null) is N*a.
            and the standard deviation is sqrt(N*a(1-a))
        :param n_alpha: no of records less than alpha
        :param no_records: no of records
        :param n_alpha: alpha threshold
        """

        # nds = np.array(range(0, valpha.shape[0])) #count over these
        score = np.zeros(alpha.shape[0])
        inds = n_alpha > no_records * alpha
        score[inds] = np.true_divide(
            n_alpha[inds] - no_records[inds] * alpha[inds],
            np.sqrt(no_records[inds] * alpha[inds] * (1.0 - alpha[inds])),
        )
        return score
