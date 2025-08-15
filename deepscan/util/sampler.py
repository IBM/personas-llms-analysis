"""Sampler"""

import numpy as np


class Sampler:
    """Samples randomly from the clean/anom pvalue ranges"""

    @staticmethod
    def sample(
        records_pvalue_ranges,
        anom_records_pvalue_ranges,
        sample_data_size,
        sample_anom_data_size,
        run,
        conditional=False,
    ):
        """
        Samples randomly from the clean/anom pvalue ranges. One set of samples for each run
        :param records_pvalue_ranges: pvalues ranges from clean records
        :param anom_records_pvalue_ranges: pvalues ranges from anomalous records
        :param sample_data_size: number of clean pvalue ranges to sample
        :param sample_anom_data_size: number of anomalous pvalue ranges to sample
        :param run: number of runs.
        """

        if conditional:
            assert len(records_pvalue_ranges.shape) == 4
            assert records_pvalue_ranges.shape[3] == 2

        else:
            assert len(records_pvalue_ranges.shape) == 3
            assert records_pvalue_ranges.shape[2] == 2

        assert records_pvalue_ranges.shape[1] == anom_records_pvalue_ranges.shape[1]

        data_size = records_pvalue_ranges.shape[0]
        anom_data_size = anom_records_pvalue_ranges.shape[0]

        assert sample_data_size <= data_size
        assert sample_anom_data_size <= anom_data_size

        if sample_data_size == 1 and sample_anom_data_size == 0:
            return [records_pvalue_ranges[i] for i in range(run)], None

        if sample_anom_data_size == 1 and sample_data_size == 0:
            return [anom_records_pvalue_ranges[i] for i in range(run)], None

        samples = []
        sampled_indices = []
        for _ in range(run):
            np.random.seed()
            choice = np.random.choice(range(data_size), sample_data_size, replace=False)
            np.random.seed()
            anom_choice = np.random.choice(
                range(anom_data_size), sample_anom_data_size, replace=False
            )
            combined_pvalue_ranges_array = np.concatenate(
                (
                    records_pvalue_ranges[choice],
                    anom_records_pvalue_ranges[anom_choice],
                ),
                axis=0,
            )
            samples.append(combined_pvalue_ranges_array)
            sampled_indices.append((choice, anom_choice))

        return samples, sampled_indices
