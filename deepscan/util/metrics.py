"""Metrics"""


class Metrics:
    """Helper class to compute metrics from the output of the scanner"""

    def __init__(self, outputfile):
        self.outputfile = outputfile

    def get_metrics_from_run(
        self, best_score, image_sub, node_sub, optimal_alpha, clean_ssize, anom_ssize
    ):
        """
        Compute key metrics such as precision, recall, and writes those including results of scan to outputfile
        :param best_score: best score from fgss_scan
        :param image_sub: list of subsets of records that have been marked as anomalous
        :param node_sub: list of subsets of attributes/nodes that have been marked as anomalous
        :param optimal_alpha: alpha that yeilded the best score across the restarts in the scan
        :param clean_ssize: sample size used in scan of clean records
        :param clean_ssize: sample size used of anomalous records
        """
        # assert(clean_ssize > 0)
        # assert(anom_ssize > 0)

        if clean_ssize > 0 and anom_ssize > 0:
            sample_size = clean_ssize + anom_ssize

            intersection = list(
                set(range(clean_ssize, sample_size)) & set(image_sub.tolist())
            )
            intersection_size = float(len(intersection))

            precision = intersection_size / len(image_sub)
            recall = intersection_size / anom_ssize

        else:
            precision = None
            recall = None

        outfile = open(self.outputfile, "a+")
        outfile.write(
            "{} {} {} {} {} {} {} {}\n".format(
                best_score,
                precision,
                recall,
                len(image_sub),
                len(node_sub),
                optimal_alpha,
                ",".join(str(x) for x in node_sub),
                ",".join(str(x) for x in image_sub),
            )
        )
        outfile.close()
