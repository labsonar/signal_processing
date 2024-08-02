"""Module for handling access to processed data from audio based in a folder.

This module defines a class, `AudioFileProcessor`, which facilitates access to processed data by
providing methods for loading and retrieving dataframes based on specified parameters. It supports
the processing and normalization of audio data, allowing users to work with either window-based or
image-based input types.
"""
import os
import enum
import typing
import json
import hashlib

import tqdm
import PIL
import pandas as pd
import numpy as np

import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt
import matplotlib.colors as color
import tikzplotlib as tikz

import lps_utils.prefered_number as lps_pn
import lps_sp.signal as lps_signal
import lps_sp.acoustical.analysis as lps_analysis


class PlotType(enum.Enum):
    """Enum defining plot types."""
    SHOW_FIGURE = 0
    EXPORT_RAW = 1
    EXPORT_PLOT = 2
    EXPORT_TEX = 3

    def __str__(self):
        return str(self.name).rsplit('.', maxsplit=1)[-1].lower()


class AudioFileProcessor():
    """ Class for handling acess to process data from a dataset. """

    def __init__(self,
                wav_base_dir: str,
                processed_base_dir: str,
                analysis: lps_analysis.SpectralAnalysis,
                params: lps_analysis.Parameters = lps_analysis.Parameters(),
                integration: lps_analysis.TimeIntegration = None,
                normalization: lps_signal.Normalization = lps_signal.Normalization.NORM_L2,
                frequency_limit: float = None,
                extract_id: typing.Callable[[str], int] = None) -> None:
        """
        Args:
            wav_base_dir (str): Base directory containing the .wav files.
            processed_base_dir (str): Base directory for storing processed data.
            analysis (lps_analysis.SpectralAnalysis): Spectral analysis to apply.
            params (lps_analysis.Parameters, optional): Parameters for the spectral analysis.
                Defaults to lps_analysis.Parameters().
            integration (lps_analysis.TimeIntegration, optional): Time integration parameters.
                Defaults to None.
            normalization (lps_signal.Normalization, optional): Normalization method.
                Defaults to lps_signal.Normalization.NORM_L2.
            frequency_limit (float, optional): Upper limit for frequency filtering.
                Defaults to None.
            extract_id (typing.Callable[[str], int], optional): Function to extract file ID from
                filename. Defaults to None.
        """
        self.wav_base_dir = wav_base_dir
        self.processed_base_dir = processed_base_dir
        self.analysis = analysis
        self.normalization = normalization
        self.params = params
        self.integration = integration
        self.frequency_limit = frequency_limit
        self.extract_id = extract_id
        self.fileid_dict = {}
        self.filename_dict = {}

        self._check_processed_dir()
        self._find_files()

    def _get_processed_dir(self) -> str:
        """Get the directory path for storing processed data.

        Returns:
            str: Path to the processed data directory.
        """
        return os.path.join(self.processed_base_dir,
                                  str(self.analysis) + "_" + self._get_hash())

    def _to_dict(self) -> typing.Dict:
        """Convert the processor's configuration to a dictionary.

        Returns:
            typing.Dict: Dictionary containing the configuration parameters.
        """
        return {
            'dataset': os.path.basename(os.path.normpath(self.wav_base_dir)),
            'normalization': str(self.normalization),
            'analysis': str(self.analysis),

            'n_spectral_pts': self.params.n_spectral_pts,
            'overlap': self.params.overlap,
            'n_mels': self.params.n_mels,
            'decimation_rate': self.params.decimation_rate,
            'log_scale': self.params.log_scale,

            'in_seconds': 'None' if self.integration is None else
                    self.integration.in_seconds,
            'integration_interval': 'None' if self.integration is None else
                    self.integration.integration_interval,
            'integration_overlap': 'None' if self.integration is None else
                    self.integration.integration_overlap,

            'frequency_limit': self.frequency_limit
        }

    def _get_hash(self) -> str:
        """Generate a hash of the processor's configuration.

        Returns:
            str: MD5 hash of the configuration dictionary.
        """
        converted = json.dumps(self._to_dict(), sort_keys=True)
        hash_obj = hashlib.md5(converted.encode())
        return hash_obj.hexdigest()

    def _save(self, path: str = None):
        """Save the processor's configuration to a file.

        Args:
            path (str, optional): Directory to save the configuration file. Defaults to None.
        """
        config_file = os.path.join(self._get_processed_dir() if path is None else path,
                                "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=4)

    def _check_processed_dir(self) -> None:
        """Check if the processed data directory exists, and create it if necessary."""
        if not os.path.exists(self._get_processed_dir()):
            os.makedirs(self._get_processed_dir(), exist_ok=True)
            self._save()

    def _find_files(self) -> None:
        """Find and index all .wav files in the base directory."""
        fileid_dict = {}
        for root, _, files in os.walk(self.wav_base_dir):
            for file in files:
                _, extension = os.path.splitext(file)
                if extension == ".wav":
                    abs_filename = os.path.join(root, file)
                    if self.extract_id is None:
                        fileid = len(fileid_dict)
                    else:
                        fileid = self.extract_id(abs_filename)

                    fileid_dict[fileid] = abs_filename

        sorted_fileids = sorted(fileid_dict.keys())

        #sorted dictionaries
        self.fileid_dict = {fileid: fileid_dict[fileid] for fileid in sorted_fileids}
        self.filename_dict = {fileid_dict[fileid]: fileid for fileid in sorted_fileids}

    def get_filenames(self) -> typing.List[str]:
        """Get a list of all filenames.

        Returns:
            typing.List[str]: List of filenames.
        """
        return list(self.filename_dict.keys())

    def get_fileids(self) -> typing.List[int]:
        """Get a list of all file IDs.

        Returns:
            typing.List[int]: List of file IDs.
        """
        return list(self.fileid_dict.keys())

    def _to_fileid(self, file: typing.Union[int, str]) -> int:
        """Convert a filename or file ID to a file ID.

        Args:
            file (typing.Union[int, str]): Filename or file ID.

        Returns:
            int: File ID.

        Raises:
            FileNotFoundError: If the file is not found in the base directory.
        """
        if isinstance(file, int):

            if file not in self.fileid_dict:
                raise FileNotFoundError(f'{file} not found in {self.wav_base_dir}')

            return file

        if file not in self.filename_dict:
            raise FileNotFoundError(f'{file} not found in {self.wav_base_dir}')

        return self.filename_dict[file]

    def process(self, file: typing.Union[int, str]) -> typing.Tuple[np.array, np.array, np.array]:
        """Process the specified file and return the power spectrum, frequencies, and times.

        Args:
            file (typing.Union[int, str]): Filename or file ID.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: Power spectrum, frequencies, and times.
        """
        file_id = self._to_fileid(file)

        processed_filename = os.path.join(self._get_processed_dir(), f'{file_id}.pkl')

        if os.path.exists(processed_filename):
            data_dict = pd.read_pickle(processed_filename)
            return data_dict['power'], data_dict['freqs'], data_dict['times']


        filename = self.fileid_dict[file_id]

        fs, data = scipy_wav.read(filename)

        # processing only first channel
        if data.ndim != 1:
            data = data[:,0]

        power, freqs, times = self.analysis.apply(data = data,
                                                  fs = fs,
                                                  params=self.params)

        if self.integration is not None:
            power, freqs, times = self.integration.apply(power, freqs, times)

        if self.frequency_limit:
            index_limit = next((i for i, freq in enumerate(freqs)
                                if freq > self.frequency_limit), len(freqs))
            freqs = freqs[:index_limit]
            power = power[:index_limit,:]

        if self.normalization is not None:
            power = self.normalization(power)

        data_to_save = {'power': power, 'freqs': freqs, 'times': times}
        pd.to_pickle(data_to_save, processed_filename)

        return power, freqs, times

    def file_to_df(self, file: typing.Union[int, str]) -> typing.Tuple[pd.DataFrame, np.array]:
        """Convert the processed file data to a DataFrame.

        Args:
            file (typing.Union[int, str]): Filename or file ID.

        Returns:
            typing.Tuple[pd.DataFrame, np.array]: DataFrame of power spectrum and array of times.
        """
        file_id = self._to_fileid(file)

        power, freqs, times = self.process(file_id)
        columns = [f'f {i}' for i in range(len(freqs))]
        power_df = pd.DataFrame(power.T, columns=columns)
        return power_df, times

    def plot(self,
             file: typing.Union[typing.Union[int, str], typing.Iterable[typing.Union[int, str]]],
             plot_type: PlotType = PlotType.EXPORT_PLOT,
             frequency_in_x_axis: bool=False,
             colormap: color.Colormap = plt.get_cmap('jet'),
             override: bool = False) -> None:
        """
        Display or save images with processed data.

        Parameters:
            file (Union[int, str, Iterable[Union[int, str]]]): ID or list of IDs of the file(s)
                to plot.
            plot_type (PlotType): Type of plot to generate. Default is PlotType.EXPORT_PLOT.
            frequency_in_x_axis (bool): If True, plot frequency values on the x-axis.
                Default: False.
            colormap (Colormap): Colormap to use for the plot. Default: 'jet'.
            override (bool): If True, override any existing saved plots. Default: False.

        Returns:
            None
        """
        if plot_type != PlotType.SHOW_FIGURE:
            output_dir = os.path.join(self._get_processed_dir(), str(plot_type))
            os.makedirs(output_dir, exist_ok=True)

        if isinstance(file, list):
            for loop_file in tqdm.tqdm(file, desc='Plot', leave=False):
                self.plot(
                    file = loop_file,
                    plot_type = plot_type,
                    frequency_in_x_axis = frequency_in_x_axis,
                    colormap = colormap,
                    override = override)
            return

        file_id = self._to_fileid(file)

        if plot_type == PlotType.EXPORT_RAW or plot_type == PlotType.EXPORT_PLOT:
            filename = os.path.join(output_dir,f'{file_id}.png')
        elif plot_type == PlotType.EXPORT_TEX:
            filename = os.path.join(output_dir,f'{file_id}.tex')
        else:
            filename = " "

        if os.path.exists(filename) and not override:
            return

        power, freqs, times = self.process(file_id)

        if frequency_in_x_axis:
            power = power.T

        if plot_type == PlotType.EXPORT_RAW:
            power = colormap(power)
            power_color = (power * 255).astype(np.uint8)
            image = PIL.Image.fromarray(power_color)
            image.save(filename)
            return

        n_ticks = 5
        time_labels = [lps_pn.get_engineering_notation(times[i], "s")
                    for i in np.linspace(0, len(times)-1, num=n_ticks, dtype=int)]

        frequency_labels = [lps_pn.get_engineering_notation(freqs[i], "Hz")
                    for i in np.linspace(0, len(freqs)-1, num=n_ticks, dtype=int)]

        time_ticks = [(x/4 * (len(times)-1)) for x in range(n_ticks)]
        frequency_ticks = [(y/4 * (len(freqs)-1)) for y in range(n_ticks)]

        plt.figure()
        plt.imshow(power, aspect='auto', origin='lower', cmap=colormap)
        plt.colorbar()

        if frequency_in_x_axis:
            plt.ylabel('Time')
            plt.xlabel('Frequency')
            plt.yticks(time_ticks)
            plt.gca().set_yticklabels(time_labels)
            plt.xticks(frequency_ticks)
            plt.gca().set_xticklabels(frequency_labels)
            plt.gca().invert_yaxis()
        else:
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.xticks(time_ticks)
            plt.gca().set_xticklabels(time_labels)
            plt.yticks(frequency_ticks)
            plt.gca().set_yticklabels(frequency_labels)

        plt.tight_layout()

        if plot_type == PlotType.SHOW_FIGURE:
            plt.show()
        elif plot_type == PlotType.EXPORT_PLOT:
            plt.savefig(filename)
            plt.close()
        elif plot_type == PlotType.EXPORT_TEX:
            tikz.save(filename)
            plt.close()

    def files_to_df(self,
               file_ids: typing.Iterable[typing.Union[int, str]],
               targets: typing.Iterable) -> typing.Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieve data for the given file IDs.

        Parameters:
            file_ids (Iterable[Union[int, str]]): The list of IDs or filenames to fetch data for.
            targets (Iterable): List of target values corresponding to the file IDs.
                Should have the same number of elements as file_ids.

        Returns:
            Tuple[pd.DataFrame, pd.Series]:
                - pd.DataFrame: The DataFrame containing the processed data.
                - pd.Series: The Series containing the target values, with the same type
                    as the target input.
        """
        result_df = pd.DataFrame()
        result_target = pd.Series()

        for file_id, target in tqdm.tqdm(
                                list(zip(file_ids, targets)), desc='Getting data', leave=False):
            data_df, _ = self.file_to_df(file_id)
            result_df = pd.concat([result_df, data_df], ignore_index=True)

            replicated_targets = pd.Series([target] * len(data_df), name='Target')
            result_target = pd.concat([result_target, replicated_targets], ignore_index=True)

        return result_df, result_target

    def __eq__(self, other: object) -> bool:
        """
        Check if two AudioFileProcessor instances are equal.

        Parameters:
            other (object): The other instance to compare with.

        Returns:
            bool: True if the instances are equal, False otherwise.
        """
        if isinstance(other, AudioFileProcessor):
            return self._get_hash() == other._get_hash()
        return False
