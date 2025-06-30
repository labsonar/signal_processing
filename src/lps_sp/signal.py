"""
Signal Description Module

This module contains signal processing common methods.
"""
import enum
import typing

import numpy as np
import scipy.signal as scipy
import scipy.io.wavfile as wavfile

import lps_utils.quantities as lps_qty


def decimate(data: np.array, decimation_rate: typing.Union[int, float]) -> np.array:
    """
    Decimate the input data by the given decimation rate.

    Args:
        data (np.ndarray): Input data.
        decimation_rate (typing.Union[int, float]): Decimation rate.
            Supports both integer and float values (equivalent to scipy.decimate if int)

    Returns:
        np.ndarray: Decimated data.
    """
    if decimation_rate == 1:
        return data

    b, a = scipy.cheby1(8, 0.05, 0.8 / decimation_rate, btype='low')
    y = scipy.filtfilt(b, a, data)
    return scipy.resample(y, int(len(y) / decimation_rate))

def tpsw(data: np.array, n: int = None, p: int = None, a: int = None) -> np.array:
    """Perform TPSW data calculation

    Args:
        data (np.array): data to process
        n (int, optional): TPSW n parameter, number of ones sample on each side zero filter.
            Defaults to None(int(round(n_pts*.04/2.0+1))).
        p (int, optional): TPSW p parameter, number of zeros from central sample.
            Defaults to None(int(round(n / 8.0 + 1)))
        a (int, optional): TPSW a parameter, threshold to saturate in the first pass filter.
            Defaults to None(2.0).

    Returns:
        np.array: TPSW output - background signal
    """

    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_pts = data.shape[0]

    if n is None:
        n = int(round(n_pts * .04 / 2.0 + 1))
    if p is None:
        p = int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p>0:
        h = np.concatenate((np.ones((n - p + 1)), np.zeros(2 * p - 1), np.ones((n - p + 1))))
    else:
        h = np.ones((1, 2 * n + 1))
        p = 1

    h /= np.linalg.norm(h, 1)
    def apply_on_spectre(xs):
        return scipy.convolve(h, xs, mode='full')
    mx = np.apply_along_axis(apply_on_spectre, arr=data, axis=0)

    ix = int(np.floor((h.shape[0] + 1)/2.0))
    mx = mx[ix-1:n_pts+ix-1]
    ixp = ix - p
    mult = 2 * ixp / \
        np.concatenate([np.ones(p-1) * ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis]
    mx[:ix,:] = mx[:ix,:] * (np.matmul(mult, np.ones((1, data.shape[1]))))
    mx[n_pts-ix:n_pts,:] = mx[n_pts-ix:n_pts,:] * \
        np.matmul(np.flipud(mult),np.ones((1, data.shape[1])))

    indl = (data-a*mx) > 0
    data = np.where(indl, mx, data)
    mx = np.apply_along_axis(apply_on_spectre, arr=data, axis=0)
    mx = mx[ix-1:n_pts+ix-1,:]
    mx[:ix,:] = mx[:ix,:] * (np.matmul(mult,np.ones((1, data.shape[1]))))
    mx[n_pts-ix:n_pts,:] = mx[n_pts-ix:n_pts,:] * \
        (np.matmul(np.flipud(mult),np.ones((1,data.shape[1]))))
    return mx

class Normalization(enum.Enum):
    """ Enum class representing the available normalizations in this module. """
    MIN_MAX = 0
    MIN_MAX_ZERO_CENTERED = 1
    NORM_L1 = 2
    NORM_L2 = 3
    NONE = 4

    def apply(self, data: np.array) -> np.array:
        """
        Apply normalization to input data. Equivalent to __cal__

        Args:
            data (np.array): The data to be normalized.

        Raises:
            NotImplementedError: Raised when the specific normalization method is not
                implemented in this module.

        Returns:
            np.array: The normalized data.
        """
        if self == Normalization.MIN_MAX:
            return (data - np.min(data, axis=0))/(np.max(data, axis=0) - np.min(data, axis=0))

        if self == Normalization.MIN_MAX_ZERO_CENTERED:
            return data/np.max(np.abs(data), axis=0)

        if self == Normalization.NORM_L1:
            # to ensure that the data is positive to avoid negative results
            data = Normalization.MIN_MAX.apply(data)
            return data/np.linalg.norm(data, ord=1, axis=0)

        if self == Normalization.NORM_L2:
            # to ensure that the data is positive to avoid negative results
            data = Normalization.MIN_MAX.apply(data)
            return data/np.linalg.norm(data, ord=2, axis=0)

        if self == Normalization.NONE:
            return data

        raise NotImplementedError(f"{self} not implemented")

    def __call__(self, data: np.array) -> np.array:
        """ Function to implicitly normalize data by calling apply. """
        return self.apply(data)

    def __str__(self):
        return super().__str__().split(".")[-1].replace("_", " ").title()

def save_normalized_wav(signal: np.ndarray,
                        fs: typing.Union[int, lps_qty.Frequency],
                        filename: str) -> None:
    """Export a .wav file

    Args:
        signal (np.ndarray): Signal to be normalized and exported
        fs (int, lps_qty.Frequency): Sample Frequency
        filename (str): Filename
    """
    if isinstance(fs, lps_qty.Frequency):
        fs = int(fs.get_hz())
    normalized = Normalization.MIN_MAX_ZERO_CENTERED(signal)
    wav_signal = (normalized * 32767).astype(np.int16)
    wavfile.write(filename, fs, wav_signal)
