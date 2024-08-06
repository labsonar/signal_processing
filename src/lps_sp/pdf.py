"""
This module provides functionality to estimate probability density functions (PDFs) and 
compute similarity measures between them. It includes an enumeration for different similarity 
measures and methods to calculate these measures from PDFs or directly from data sets.
"""
import enum
import typing

import numpy as np
import matplotlib.pyplot as plt
import scipy


def estimate_pdf(window1: np.array, window2: np.array, n_bins: int) -> \
        typing.Tuple[np.array, np.array, np.array]:
    """
    Estimate the probability density function (PDF) for two windows of data.

    Args:
        window1 (np.array): First data window.
        window2 (np.array): Second data window.
        n_bins (int): Number of bins to use for the histogram.

    Returns:
        typing.Tuple[np.array, np.array, np.array]: Normalized histograms (PDFs) for window1 and
            window2, and the commom bin edges (one more sample than the pdfs).
    """
    min_value = np.min([np.min(window1), np.min(window2)])
    max_value = np.max([np.max(window1), np.max(window2)])
    bins = np.linspace(min_value, max_value, n_bins)

    x1_dist, edges = np.histogram(window1, bins=bins, density=True)
    x2_dist, _ = np.histogram(window2, bins=bins, density=True)

    return x1_dist, x2_dist, edges


def plot_pdf(output_filename: str, window1: np.array, window2: np.array, n_bins: int,
             title1: str = "Window 1", title2: str = "Window 2") -> None:
    """
    Plot the probability density function (PDF) for two windows of data.

    Args:
        output_filename (str): Output image filename.
        window1 (np.array): First data window.
        window2 (np.array): Second data window.
        n_bins (int): Number of bins to use for the histogram.
        title1 (str, optional): Title of the plot from window 1. Default 'Window 1'
        title2 (str, optional): Title of the plot from window 2. Default 'Window 2'
    """
    y1, y2, x = estimate_pdf(window1, window2, n_bins)

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    ax[0].bar(x[1:], y1, align="edge", width=np.diff(x))
    ax[0].set_title(f"PDF {title1}")
    ax[0].set_xlabel("Noise Magnitude")
    ax[0].set_ylabel("Probability")
    ax[0].grid(True, which='both', ls='-')

    ax[1].bar(x[1:], y2, align="edge", width=np.diff(x))
    ax[1].set_title(f"PDF {title2}")
    ax[1].set_xlabel("Noise Magnitude")
    ax[1].set_ylabel("Probability")
    ax[1].grid(True, which='both', ls='-')

    plt.savefig(output_filename)
    plt.close()

class DissimilarityMeasure(enum.Enum):
    """ Enum class for measure similarity between datas/pdfs."""
    KL_DIVERGENCE = 0
    WASSERSTEIN = 1
    JENSEN_SHANNON = 2

    def __str__(self) -> str:
        """ Return a printable string. """
        labels = ['KLdiv', 'Wasserstein', 'JSD']
        return labels[self.value]

    def from_pdf(self, pdf1, pdf2) -> float:
        """
        Calculate the similarity measure between two PDFs.

        Args:
            pdf1 (np.array): First PDF.
            pdf2 (np.array): Second PDF.

        Returns:
            float: Similarity measure between pdf1 and pdf2.
        """
        if self == DissimilarityMeasure.KL_DIVERGENCE:
            pdf1 = np.where(pdf1 < 1e-10, 1e-10, pdf1)
            pdf2 = np.where(pdf2 < 1e-10, 1e-10, pdf2)
            return np.sum(scipy.special.kl_div(pdf1, pdf2))

        if self == DissimilarityMeasure.WASSERSTEIN:
            return scipy.stats.wasserstein_distance(pdf1, pdf2)

        if self == DissimilarityMeasure.JENSEN_SHANNON:
            return scipy.spatial.distance.jensenshannon(pdf1, pdf2)

        raise NotImplementedError(f"{str(self)} not implemented")

    def from_data(self, data1, data2, n_bins = 100) -> float:
        """
        Calculate the similarity measure between two data sets by first estimating their PDFs
            by histogram.

        Args:
            data1 (np.array): First data set.
            data2 (np.array): Second data set.
            n_bins (int, optional): Number of bins to use for the histogram estimation.
                Defaults to 100.

        Returns:
            float: Similarity measure between data1 and data2.
        """
        x1_dist, x2_dist, _ = estimate_pdf(data1, data2, n_bins=n_bins)
        return self.from_pdf(x1_dist, x2_dist)
