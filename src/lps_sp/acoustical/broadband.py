"""
Acoustical Signal Module

This module contains functions for generating and evaluate broadband signals.
"""
import enum
import typing
import numpy as np
import scipy.signal as scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import lps_utils.quantities as lps_qty

def generate(frequencies: np.array, psd_db: np.array, n_samples: int, fs: float,
             seed: typing.Union[int, np.random.Generator] = None,
             filter_state: np.ndarray = None) -> typing.Tuple[np.array, np.ndarray]:
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

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed = seed)
    order = 1025
    noise = rng.normal(0, std_dev, n_samples + order)
    # Generate more samples to eliminate filter transient response


    # Normalize frequencies between 0 and 1 (fs/2)
    if np.max(frequencies) > 1:
        frequencies = frequencies / (fs / 2)

    # Normalize gain for each PSD
    intensities_norm = psd_linear/np.max(psd_linear)

    if np.min(intensities_norm) == 1:
        return noise[order:], None

    coeficient = scipy.firwin2(order, frequencies, np.sqrt(intensities_norm), antisymmetric=False)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin2.html
    # antisymmetric=False, order=odd to force filter type 1,
    # in which the frequencies fs/2 and 0 must not be 0

    if filter_state is None:
        zi = scipy.lfilter_zi(coeficient, 1) * noise[0]
    else:
        zi = filter_state

    out_noise, zf = scipy.lfilter(coeficient, 1, noise, zi=zi)
    return out_noise[order:], zf

def psd(signal: np.array,
        fs: typing.Union[float, lps_qty.Frequency],
        window_size: int = 1024,
        overlap: typing.Union[int, float] = 0,
        window: str = 'hann',
        db_unity = True) -> typing.Tuple[np.array, np.array]:
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

    if isinstance(fs, lps_qty.Frequency):
        fs = fs.get_hz()

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

def plot_psd(filename: str, noise: np.array, fs: lps_qty.Frequency)-> None:
    """
    Plots and saves the Power Spectral Density (PSD) of a single noise signal.

    Parameters:
        filename (str): The path to save the resulting PSD plot image.
        noise (np.array): The input noise signal as a 1D NumPy array.
        fs (lps_qty.Frequency): Sampling frequency of the signal.
    """

    plt.figure(figsize=(10, 6))
    f_bb, i_bb = psd(noise, fs=fs)
    plt.plot(f_bb, i_bb)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_psds(filename: str,
              noises: typing.List[np.array],
              labels: typing.List[str],
              fs: lps_qty.Frequency)-> None:
    """
    Plots and saves the Power Spectral Density (PSD) curves of multiple noise signals.

    Parameters:
        filename (str): The path to save the resulting PSD plot image.
        noises (List[np.array]): A list of 1D NumPy arrays, each representing a noise signal.
        labels (List[str]): A list of labels corresponding to each noise signal.
        fs (lps_qty.Frequency): Sampling frequency of the signals.
    """

    plt.figure(figsize=(10, 6))
    cmap = cm.get_cmap("viridis", min(len(noises), len(labels)))
    for i, (noise, label) in enumerate(zip(noises, labels)):
        f_bb, i_bb = psd(noise, fs=fs.get_hz())
        plt.plot(f_bb, i_bb, label=label, color=cmap(i))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class ColoredNoises(enum.Enum):
    """ Enum class to represent and generate colored noises. """
    WHITE = 0
    PINK = 1
    BROWN = 2
    BLUE = 3
    VIOLET = 4

    def __str__(self):
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower() + "_noise"

    def to_psd(self, fs: float = 48000, ref_db: float = 50):
        """
        Get the Power Spectral Density (PSD) for the selected colored noise.

        Args:
            fs (float, optional): The sampling frequency in Hz. Defaults to 48,000 Hz.
            ref_db (float, optional): The reference level in dB. Defaults to 50 dB.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Frequencies in Hz.
                - PSD estimates in dB.

        Raises:
            ValueError: If an unsupported noise type is selected.
        """

        ref_freq = 1
        frequencies = np.logspace(np.log10(ref_freq), np.log10(fs / 2), num=100)

        if self == ColoredNoises.WHITE:
            gain_per_octave = 0
        elif self == ColoredNoises.PINK:
            gain_per_octave = -3
        elif self == ColoredNoises.BROWN:
            gain_per_octave = -6
        elif self == ColoredNoises.BLUE:
            gain_per_octave = +3
        elif self == ColoredNoises.VIOLET:
            gain_per_octave = +6

        else:
            raise NotImplementedError(f"Unsupported noise type: {self}")

        intensities = ref_db + gain_per_octave * np.log2(frequencies/ref_freq)

        return frequencies, intensities

    def generate(self, n_samples: int, fs: float = 48000, ref_db: float = 50):
        """
        Get samples for the selected colored noise.

        Args:
            n_samples (int): Number of samples to generate.
            fs (float, optional): The sampling frequency in Hz. Defaults to 48,000 Hz.
            ref_db (float, optional): The reference level in dB. Defaults to 50 dB.

        Returns:
            np.array: Generated broadband noise in μPa.
        """
        frequencies, intensities = self.to_psd(fs=fs, ref_db=ref_db)
        return generate(frequencies=frequencies, psd_db=intensities, n_samples=n_samples, fs=fs)
