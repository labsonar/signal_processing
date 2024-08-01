"""
Spectral Analysis Script

This script reads an audio file, applies the specified type of spectral analysis,
    and saves the result as a PNG image.

Usage:
    The script can be run from the command line with the following arguments:
        analysis (str): The type of spectral analysis to perform.
            Choices are ['spectrogram', 'lofar', 'melgram'].
        filename (str): The path to the audio file to analyze.

Example:
    python script_name.py spectrogram path/to/audio.wav

    will be generated the image path/to/audio_spectrogram.png
"""
import argparse
import os

import numpy as np
import scipy.io.wavfile as scipy_wav
import matplotlib.pyplot as plt

import lps_sp.acoustical_analysis as lps_analysis


def main(analysis: lps_analysis.SpectralAnalysis,
         filename: str,
         params: lps_analysis.Parameters = lps_analysis.Parameters()) -> None:
    """
    Perform spectral analysis on an audio file and save the result as an image.

    Args:
        analysis (lps_analysis.SpectralAnalysis): The type of spectral analysis to perform.
        filename (str): The path to the audio file to analyze.
        params (lps_analysis.Parameters, optional): Parameters for the spectral analysis.
            Defaults to lps_analysis.Parameters().
    """

    fs, data = scipy_wav.read(filename)

    power, freq, time = analysis.apply(data=data, fs=fs, params=params)

    output_file = os.path.splitext(filename)[0] + f'_{analysis}.png'

    plt.figure()
    plt.imshow(power, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               extent=[time[0], time[-1], freq[0], freq[-1]])
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":

    choises = [str(i) for i in lps_analysis.SpectralAnalysis]

    parser = argparse.ArgumentParser(description='Run spectral analysis on an audio file')
    parser.add_argument('analysis', type=str, choices=choises,
                        help='Type of spectral analysis to perform')
    parser.add_argument('filename', type=str,
                        help='Path to the audio file to analyze')

    parser.add_argument('--n_spectral_pts', type=int, default=1024,
                        help='Number of points for spectral analysis (default: 1024)')
    parser.add_argument('--overlap', type=float, default=0,
                        help='Number of points or window factor to overlap (default: 0)')
    parser.add_argument('--n_mels', type=int, default=256,
                        help='Number of Mel bands (default: 256)')
    parser.add_argument('--decimation_rate', type=float, default=1,
                        help='Decimation rate (default: 1)')
    parser.add_argument('--log_scale', type=bool, default=True,
                        help='Apply log scale to the result (default: True)')


    args = parser.parse_args()

    main(analysis = lps_analysis.SpectralAnalysis(choises.index(args.analysis)),
        filename = args.filename,
        params = lps_analysis.Parameters(
            n_spectral_pts=args.n_spectral_pts,
            overlap=args.overlap,
            n_mels=args.n_mels,
            decimation_rate=args.decimation_rate,
            log_scale=args.log_scale
        )

)
