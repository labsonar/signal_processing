"""
Analysis Description Module

This module contains a class and functions for applying spectral analysis to input data.
"""
import enum
import typing

import numpy as np
import scipy.signal as scipy
import librosa

import lps_sp.signal as lps_signal


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
