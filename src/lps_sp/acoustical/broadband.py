"""
Acoustical Signal Module

This module contains functions for generating and evaluate broadband signals.
"""
import typing
import numpy as np
import scipy.signal as scipy


def generate(frequencies: np.array, psd_db: np.array, n_samples: int, fs: float) -> np.array:
    """Generate broadband noise based on frequency and intensity information.

    Args:
        frequencies (np.array): Array of frequency values.
        psd_db (np.array): Array of Power Spectral Density (PSD) values in dB ref 1μPa @1m/√Hz.
        n_samples (int): Number of samples to generate.
        fs (float): Sampling frequency.

    Returns:
        np.array: Generated broadband noise in μPa.

    Raises:
        UnboundLocalError: Raised if frequencies and intensities have different lengths.

    """

    if len(frequencies) != len(psd_db):
        raise UnboundLocalError("for generate_noise frequencies and "\
                                "intensities must have the same length")

    # Ensure frequencies include 0 and fs/2
    # (required by scipy.firwin2) and are within the Nyquist limit
    index = np.argmax(frequencies > (fs / 2.0))
    if index > 0:
        if frequencies[index - 1] == (fs / 2):
            frequencies = frequencies[:index]
            psd_db = psd_db[:index]
        else:

            lf_min = np.log10(frequencies[index - 1])
            lf_h = np.log10(fs / 2)
            lf_max = np.log10(frequencies[index])

            gain_per_decade = (psd_db[index] - psd_db[index - 1])/(lf_max-lf_min)

            i = psd_db[index - 1] + gain_per_decade * (lf_h-lf_min)

            frequencies = np.append(frequencies[:index], fs/2)
            psd_db = np.append(psd_db[:index], i)
    else:
        if frequencies[-1] != (fs / 2):

            lf_min = np.log10(frequencies[-2])
            lf_max = np.log10(frequencies[-1])
            lf_h = np.log10(fs / 2)

            gain_per_decade = (psd_db[-1] - psd_db[-2])/(lf_max-lf_min)

            i = psd_db[-1] + gain_per_decade * (lf_h-lf_max)

            frequencies = np.append(frequencies[:-1], fs/2)
            psd_db = np.append(psd_db[:-1], i)


    if frequencies[0] != 0:
        frequencies = np.append(0, frequencies)

        if psd_db[0] < psd_db[1]:
            psd_db = np.append(0, psd_db)
        else:
            psd_db = np.append(psd_db[0], psd_db)


    psd_linear = 10 ** ((psd_db) / 20)  # Convert dB to linear scale

    # Calculate total noise power with the highest PSD of the signal
    #    P = ∫ ​psd df  for white noise => P = psd * Δf = psd * fs/2
    max_power = np.max(psd_linear) * fs/2

    # Calculate the standard deviation for the calculated power
    #   power => P = E[x^2]
    #   standard deviation => std = √(var(x)) = √(E[(x-μ)^2])
    #       as mean (μ) is zero
    #       std = √P
    std_dev = np.sqrt(max_power)

    order = 1025
    noise = np.random.normal(0, std_dev, n_samples + order)
    # Generate more samples to eliminate filter transient response


    # Normalize frequencies between 0 and 1 (fs/2)
    if np.max(frequencies) > 1:
        frequencies = frequencies / (fs / 2)

    # Normalize gain for each PSD
    intensities_norm = psd_linear/np.max(psd_linear)

    if np.min(intensities_norm) == 1:
        return noise[order:]

    coeficient = scipy.firwin2(order, frequencies, np.sqrt(intensities_norm), antisymmetric=False)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html
    # antisymmetric=False, order=odd to force filter type 1,
    # in which the frequencies fs/2 and 0 must not be 0

    out_noise = scipy.lfilter(coeficient, 1, noise)
    return out_noise[order:]

def psd(signal: np.array, fs: float, window_size: int = 1024, overlap: typing.Union[int, float] = 0,
        window: str = 'hann', db_unity = True) -> typing.Tuple[np.array, np.array]:
    """Estimate the power spectrum density (PSD) for input signal.

    Args:
        signal (np.array): data in some unity (μPa).
        fs (float): Sampling frequency, default frequency factor to fs.
        window_size (int): number of samples in each segment.
        overlap (typing.Union[int, float], optional): Number of points (int) or window factor to
                overlap (float between 0 and 1) for analysis. Defaults to 0.
        window (str, optional): Desired window to use. See scipy.signal.welch for details.
            Defaults to a Hann window
        db_unity (bool): If True, returns PSD in dB, else in linear scale.

    Returns:
        np.array: Frequencies in Hz.
        np.array: Estimated PSD in dB ref 1μPa @1m/√Hz (or in linear scale if db_unity is False).
    """
    # https://ieeexplore.ieee.org/document/1161901
    # http://resource.npl.co.uk/acoustics/techguides/concepts/siunits.html

    if isinstance(overlap, float):
        if overlap < 0 or overlap >= 1:
            raise UnboundLocalError("Overlap expected as a float in interval [0, 1[")

        overlap = int(window_size * overlap)

    freqs, intensity = scipy.welch(x=signal,
                                fs=fs,
                                window=window,
                                nperseg=window_size,
                                noverlap=overlap,
                                scaling='density',
                                axis=-1,
                                average='mean')

    if db_unity:
        intensity = 20 * np.log10(intensity)

    # Removing DC component
    return freqs[1:], intensity[1:]
