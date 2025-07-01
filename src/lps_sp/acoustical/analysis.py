"""
Analysis Description Module

This module contains a class and functions for applying spectral analysis to input data.
"""
import enum
import typing

import numpy as np
import scipy.signal as scipy
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as color

import lps_sp.signal as lps_signal
import lps_utils.prefered_number as lps_pn


class Parameters():
    """ Class for unify acoustical processing parameters. """

    def __init__(self,
                n_spectral_pts: int = 1024,
                overlap: typing.Union[int, float] = int(0),
                n_mels: int = 256,
                decimation_rate: typing.Union[int, float] = int(1),
                log_scale: bool = True) -> None:
        """
        Args:
            n_spectral_pts (int, optional): Number of points after the spectral analysis.
                Defaults to 1024.
            overlap (typing.Union[int, float], optional): Number of points or window factor to
                overlap for analysis. Defaults to 0.
            n_mels (int, optional): Number of mel bands to generate for MEL based analysis.
                Defaults to 256.
        decimation_rate (typing.Union[int, float]): Decimation rate. Supports both integer and
            float values (equivalent to scipy.decimate if int)
        log_scale (bool, optional): Apply log scale on result. Defaults to True.
        """
        self.n_spectral_pts = n_spectral_pts
        self.overlap = overlap
        self.n_mels = n_mels
        self.decimation_rate = decimation_rate
        self.log_scale = log_scale

    def scale(self, data: np.array) -> np.ndarray:
        """
        Apply log scaling to the data.

        Args:
            data (np.ndarray): Data to be scaled.

        Returns:
            np.ndarray: Scaled data.
        """
        data[data < 1e-9] = 1e-9
        data = 20*np.log10(data)
        return data

    def decimate(self, data: np.array, fs: float) -> typing.Tuple[np.array, float]:
        """
        Decimate the input data.

        Args:
            data (np.ndarray): Input data.
            fs (float): Sampling frequency.

        Returns:
            typing.Tuple[np.ndarray, float]: Decimated data and new sampling frequency.
        """
        return lps_signal.decimate(data, self.decimation_rate), fs/self.decimation_rate

    def get_fft_size(self) -> int:
        """
        Get the FFT size to generate the n_spectral_pts desired.

        Returns:
            int: FFT size.
        """
        return self.n_spectral_pts * 2

    def get_overlap(self) -> int:
        """
        Get the overlap size.

        Returns:
            int: Overlap size.
        """
        if self.overlap < 1:
            return np.floor(self.get_fft_size() * self.overlap)

        return self.overlap * 2

class TimeIntegration():
    """ Class for applying time integration to a power spectrum over a given interval. """

    def __init__(self,
                 in_seconds: bool,
                integration_interval: typing.Union[int, float],
                integration_overlap: typing.Union[int, float] = 0) -> None:
        """
        Args:
            in_seconds (bool): If True, `integration_interval` and `integration_overlap` are
                interpreted as seconds. Othewise, they are interpreted as number of samples.
            integration_interval (float): Interval for integration.
            integration_overlap (float, optional): Overlap between intervals. Defaults to 0.
        """
        self.in_seconds = in_seconds
        self.integration_interval = integration_interval
        self.integration_overlap = integration_overlap

    @staticmethod
    def from_samples(integration_interval_samples: int,
                     integration_overlap_samples: int = 0) -> 'TimeIntegration':
        """
        Named constructor for creating a TimeIntegration instance with sample-based parameters.

        Args:
            integration_interval_samples (int, optional): Interval in samples for integration.
            integration_overlap_samples (int, optional): Overlap in samples between intervals.
                Defaults to 0.

        Returns:
            TimeIntegration: An instance of TimeIntegration initialized with sample-based
                parameters.
        """
        return TimeIntegration(in_seconds = True,
                               integration_interval = integration_interval_samples,
                               integration_overlap = integration_overlap_samples)

    @staticmethod
    def from_seconds(integration_interval: float,
                     integration_overlap: float = 0) -> 'TimeIntegration':
        """
        Named constructor for creating a TimeIntegration instance with time-based parameters.

        Args:
            integration_interval (float, optional): Interval in seconds for integration.
            integration_overlap (float, optional): Overlap in seconds between intervals.
                Defaults to 0.

        Returns:
            TimeIntegration: An instance of TimeIntegration initialized with time-based parameters.
        """
        return TimeIntegration(in_seconds = False,
                               integration_interval = integration_interval,
                               integration_overlap = integration_overlap)

    def apply(self, power: np.array, freq: np.array, time: np.array) -> \
            typing.Tuple[np.array, np.array, np.array]:
        """
        Apply time integration to the power spectrum.

        Args:
            power (np.array): 2D array of power spectrum data with shape (n_freqs, n_times).
            freq (np.array): 1D array of frequency values corresponding to the rows of `power`.
            time (np.array): 1D array of time values corresponding to the columns of `power`.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: A tuple containing:
                - 2D array of integrated power spectrum with shape (n_freqs, n_final_times).
                - 1D array of frequency values (same as input).
                - 1D array of integrated time values.
        """
        delta_t = time[-1] - time[-2]
        if self.in_seconds:
            n_means = int(np.round(self.integration_interval / delta_t))
            n_overlap = int(np.round((self.integration_interval - self.integration_overlap)/ delta_t))
        else:
            n_means = self.integration_interval
            n_overlap = self.integration_overlap

        final_power = []
        final_times = []

        for i in range(0, len(time), n_overlap):
            mean_spectrum = np.mean(power[:, i:i+n_means], axis=1)
            final_power.append(mean_spectrum)
            final_times.append(time[i])

        return np.array(final_power).T, np.array(freq), np.array(final_times)


class SpectralAnalysis(enum.Enum):
    """ Enum class to represent and process the available spectral analyzes in this module """
    SPECTROGRAM = 0
    LOFAR = 1
    MELGRAM = 2

    def __str__(self):
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()

    def apply(self, data: np.array, fs: float,
              params: Parameters = Parameters()):
        """
        Perform data spectral analysis

        Args:
            - data (np.array): Input data for analysis.
            - fs (float): Sampling frequency.
            - parameters (AcousticalParameters, optional): acoustical parameters.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
        """
        return getattr(self.__class__, str(self))(data, fs, params)

    @staticmethod
    def spectrogram(data: np.array, fs: float, params: Parameters = Parameters()) -> \
            typing.Tuple[np.array, np.array, np.array]:
        """
        Perform spectrogram data analysis.

        Args:
            data (np.ndarray): Input data for analysis.
            fs (float): Sampling frequency.
            params (Parameters, optional): Acoustical parameters. Defaults to Parameters().

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
        """

        data, fs = params.decimate(data, fs)
        data = data - np.mean(data)

        n_pts = params.get_fft_size()
        n_overlap = params.get_overlap()

        freq, time, power = scipy.spectrogram(data,
                                        nfft=n_pts,
                                        fs=fs,
                                        window=np.hanning(n_pts),
                                        noverlap=n_overlap,
                                        detrend=False,
                                        scaling='spectrum',
                                        mode='complex')
        power = np.abs(power)*n_pts/2

        power = params.scale(power) # apply log scale if necessary

        # removing last sample - scipy.spectrogram return n_pts/2+1 incluing frequency 0 and fs/2
        # typically for matlab and fft even in python
        # so used n_pts/2 from frequency 0 to last frequency < fs/2
        power = power[:-1,:]
        freq = freq[:-1]
        return power, freq, time

    @staticmethod
    def lofar(data: np.array, fs: float, params: Parameters = Parameters()) -> \
            typing.Tuple[np.array, np.array, np.array]:
        """
        Perform LOFAR data analysis.

        Args:
            data (np.ndarray): Input data for analysis.
            fs (float): Sampling frequency.
            params (Parameters, optional): Acoustical parameters. Defaults to Parameters().

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
        """

        power, freq, time = SpectralAnalysis.spectrogram(data, fs, params)
        power = power - lps_signal.tpsw(power)
        power[power < -0.2] = 0
        return power, freq, time

    @staticmethod
    def melgram(data: np.array, fs: float, params: Parameters = Parameters()) -> \
                typing.Tuple[np.array, np.array, np.array]:
        """
        Perform MEL spectrogram data analysis.

        Args:
            data (np.ndarray): Input data for analysis.
            fs (float): Sampling frequency.
            params (Parameters, optional): Acoustical parameters. Defaults to Parameters().

        Returns:
            typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - 2D array representing power spectrum.
                - 1D array with output frequencies.
                - 1D array with relative time to sample 0 of the data.
        """
        data, fs = params.decimate(data, fs)
        data = data - np.mean(data)

        n_pts = params.get_fft_size()
        n_overlap = params.get_overlap()
        hop_length = int(n_pts - n_overlap)
        discard = int(np.floor(n_pts/hop_length))

        n_data = lps_signal.Normalization.MIN_MAX_ZERO_CENTERED(data).astype(float)
        power = librosa.feature.melspectrogram(
                        y=n_data,
                        sr=fs,
                        n_fft=n_pts,
                        hop_length=hop_length,
                        win_length=n_pts,
                        window=np.hanning(n_pts),
                        n_mels=params.n_mels,
                        power=2,
                        fmax=fs/2)
        if params.log_scale:
            power = librosa.power_to_db(power, ref=np.max)

        power = power[:,discard:]
        freqs = librosa.core.mel_frequencies(n_mels=params.n_mels, fmin=0.0, fmax=fs/2)

        start_time = n_pts/fs
        step_time = hop_length/fs
        times = [start_time + step_time * valor for valor in range(power.shape[1])]
        return power, freqs, times

    def plot(self,
             filename: str,
             data: np.array,
             fs: float,
             params: Parameters = Parameters(),
             integration: TimeIntegration = None,
             normalization: lps_signal.Normalization = lps_signal.Normalization.NORM_L2,
             frequency_limit: float = None,
             frequency_in_x_axis: bool = False,
             colormap: color.Colormap = plt.get_cmap('jet')):
        """
        Process data with spectral analysis and plot the power spectrum.

        Args:
            data (np.array): Input time-domain signal.
            fs (float): Sampling frequency.
            params (Parameters): Parameters for spectral analysis.
            integration (TimeIntegration, optional): Time integration object. Defaults to None.
            normalization (Normalization, optional): Normalization function. Defaults to NORM_L2.
            frequency_limit (float, optional): Max frequency to display. Defaults to None.
            frequency_in_x_axis (bool, optional): If True, frequency is x-axis, time y-axis.
                Defaults to False.
            colormap (Colormap, optional): Matplotlib colormap for the image. Defaults to 'jet'.

        Returns:
            None
        """
        power, freqs, times = self.apply(data, fs, params)

        if integration is not None:
            power, freqs, times = integration.apply(power, freqs, times)

        if frequency_limit is not None:
            index_limit = next((i for i, freq in enumerate(freqs) if freq > frequency_limit),
                               len(freqs))
            freqs = freqs[:index_limit]
            power = power[:index_limit, :]

        if normalization is not None:
            power = normalization(power)

        if frequency_in_x_axis:
            power = power.T

        n_ticks = 5
        time_labels = [lps_pn.get_engineering_notation(times[i], "s")
                    for i in np.linspace(0, len(times) - 1, num=n_ticks, dtype=int)]
        frequency_labels = [lps_pn.get_engineering_notation(freqs[i], "Hz")
                            for i in np.linspace(0, len(freqs) - 1, num=n_ticks, dtype=int)]

        time_ticks = [(x / 4 * (len(times) - 1)) for x in range(n_ticks)]
        frequency_ticks = [(y / 4 * (len(freqs) - 1)) for y in range(n_ticks)]

        plt.figure()
        plt.imshow(power, aspect='auto', origin='lower', cmap=colormap)
        plt.colorbar()

        if frequency_in_x_axis:
            plt.ylabel('Time (s)')
            plt.xlabel('Frequency (Hz)')
            plt.yticks(time_ticks)
            plt.gca().set_yticklabels(time_labels)
            plt.xticks(frequency_ticks)
            plt.gca().set_xticklabels(frequency_labels)
            plt.gca().invert_yaxis()
        else:
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.xticks(time_ticks)
            plt.gca().set_xticklabels(time_labels)
            plt.yticks(frequency_ticks)
            plt.gca().set_yticklabels(frequency_labels)

        plt.tight_layout()
        plt.savefig(filename)
