import os
import argparse
import typing
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

import lps_utils.utils as lps_utils
import lps_sp.acoustical.analysis as lps_analysis


def _resolve_input_files(input_path: str) -> typing.List[str]:
    if os.path.isfile(input_path) and input_path.endswith(".wav"):
        return [input_path]
    elif os.path.isdir(input_path):
        return lps_utils.find_files(input_path, extension=".wav")
    elif "," in input_path:
        return [f.strip() for f in input_path.split(",") if f.strip().endswith(".wav")]
    else:
        raise ValueError(f"Input path '{input_path}' is not valid.")

def _export_path(input_file: str, export_tex: bool) -> str:
    directory, filename = os.path.split(input_file)
    base, _ = os.path.splitext(filename)
    extension = ".tex" if export_tex else ".png"
    return os.path.join(directory, base + extension)

def main():
    """App main's function. """

    parser = argparse.ArgumentParser(description="Spectral analysis of .wav files.")
    parser.add_argument("input", type=str,
                        help="Input .wav file, comma-separated list, or directory")
    parser.add_argument("--n_fft", type=int, default=2048,
                        help="FFT points (default: 1024)")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Overlap (float 0-1 or int)")
    parser.add_argument("--n_mels", type=int, default=256,
                        help="Number of mel bands")
    parser.add_argument("--decimation", type=float, default=1.0,
                        help="Decimation rate")
    parser.add_argument("--linear", action="store_true",
                        help="Use linear scale")
    parser.add_argument("--integration_time", type=float, default=1.0,
                        help="Time integration window (in seconds)")
    parser.add_argument("--integration_overlap", type=float, default=0.5,
                        help="Time integration overlap (in seconds)")
    parser.add_argument("--export_tex", action="store_true",
                        help="Export figure to .tex")

    analysis_choices = [a.name.lower() for a in lps_analysis.SpectralAnalysis]
    default_analysis_str = lps_analysis.SpectralAnalysis.LOFAR.name.lower()
    parser.add_argument("--analysis", type=str, choices=analysis_choices,
                        default=default_analysis_str,
                        help="Type of spectral analysis to perform (default: lofar)")

    available_colormaps = plt.colormaps()
    default_cmap = 'jet'
    parser.add_argument("--colormap", type=str, choices=available_colormaps,
                        default=default_cmap,
                        help=f"Colormap for the plot (default: {default_cmap})")


    args = parser.parse_args()

    files = _resolve_input_files(args.input)

    params = lps_analysis.Parameters(
        n_spectral_pts=args.n_fft,
        overlap=args.overlap,
        n_mels=args.n_mels,
        decimation_rate=args.decimation,
        log_scale=not args.linear
    )

    integration = None
    if args.integration_time is not None:
        integration = lps_analysis.TimeIntegration.from_seconds(
            integration_interval=args.integration_time,
            integration_overlap=args.integration_overlap
        )

    for wav_file in tqdm.tqdm(files, desc="Processing files", ncols=120):
        fs, data = wavfile.read(wav_file)

        if data.ndim > 1:
            data = data[:, 0]

        out_path = _export_path(wav_file, args.export_tex)

        analysis_type = lps_analysis.SpectralAnalysis[args.analysis.upper()]

        analysis_type.plot(
            filename=out_path,
            data=data,
            fs=fs,
            params=params,
            integration=integration,
            frequency_in_x_axis=True,
            colormap=plt.get_cmap(args.colormap)
        )


if __name__ == "__main__":
    main()
