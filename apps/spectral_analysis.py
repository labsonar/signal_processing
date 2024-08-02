"""
Spectral Analysis Script

This script reads an audio file, applies the specified type of spectral analysis,
    and saves the result as a PNG image.

Usage:
    python script_name.py -h

Example:
    python script_name.py spectrogram path/to/audio.wav

    will be generated the image path/to/audio_spectrogram.png
"""
import argparse

import matplotlib.colors as plt_color
import matplotlib.pyplot as plt

import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.manager as lps_manager


def main(wav_input_dir: str,
         output_dir: str,
         analysis: lps_analysis.SpectralAnalysis,
         params: lps_analysis.Parameters,
         integration: lps_analysis.TimeIntegration,
         plot_type: lps_manager.PlotType,
         frequency_in_x_axis: bool,
         colormap: plt_color.Colormap,
         override: bool,
         n_files: int) -> None:
    """
    Perform spectral analysis on an audio file and save the result as an image.

    Args:
        wav_input_dir (str): Base directory to search for audio files to analyze.
        output_dir (str): Directory to save the output images.
        analysis (lps_analysis.SpectralAnalysis): The type of spectral analysis to perform.
        params (lps_analysis.Parameters): Parameters for the spectral analysis.
        integration (lps_analysis.TimeIntegration): Time integration.
        plot_type (lps_manager.PlotType): Type of plot to generate.
        frequency_in_x_axis (bool): If True, plot frequency values on the x-axis.
        colormap (plt_color.Colormap): Colormap to use for the plot.
        override (bool): If True, override any existing saved plots.
        n_files (int): Number of files to analize. -1 to process all files.
    """

    manager = lps_manager.AudioFileProcessor(
            wav_base_dir = wav_input_dir,
            processed_base_dir = output_dir,
            analysis = analysis,
            params = params,
            integration = integration,
            # extract_id = lambda filename: int(filename.rsplit('.',maxsplit=1)[0][-2:])
        )

    file_ids = manager.get_fileids()
    if n_files > 0:
        file_ids = file_ids[:n_files]

    manager.plot(
        file=file_ids,
        plot_type=plot_type,
        frequency_in_x_axis=frequency_in_x_axis,
        colormap=colormap,
        override=override)

    df, _ = manager.files_to_df(file_ids, file_ids)
    print(df)

if __name__ == "__main__":

    spectral_choises = [str(i) for i in lps_analysis.SpectralAnalysis]
    plot_choices = [str(pt) for pt in lps_manager.PlotType]

    parser = argparse.ArgumentParser(description='Run spectral analysis on an audio file')
    parser.add_argument('analysis', type=str, choices=spectral_choises,
                        help='Type of spectral analysis to perform')
    parser.add_argument('wav_input_dir', type=str,
                        help='Base directory to search audio files to analyze')

    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--n_files', type=int, default=-1,
                        help='Number of files to process')

    # lps_analysis.Parameters arguments
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

    # lps_analysis.TimeIntegration arguments
    parser.add_argument('--integration_interval', type=float, default=None,
                        help='Interval for time integration (in seconds or samples (default))')
    parser.add_argument('--integration_overlap', type=float, default=0,
                        help='Overlap for time integration (in seconds or samples (default)')
    parser.add_argument('--in_seconds', action='store_true',
                        help='Specify if the integration interval and overlap are in seconds')

    # Plotting arguments
    parser.add_argument('--plot_type', type=str, default=str(lps_manager.PlotType.EXPORT_PLOT),
            choices=plot_choices,
            help=f'Type of plot to generate (default: {str(lps_manager.PlotType.EXPORT_PLOT)})')
    parser.add_argument('--frequency_in_x_axis', action='store_true',
                        help='Plot frequency values on the x-axis')
    parser.add_argument('--colormap', type=str, default='jet',
                        help='Colormap to use for the plot (default: jet)')
    parser.add_argument('--override', action='store_true',
                        help='Override any existing saved plots')

    args = parser.parse_args()

    main(analysis = lps_analysis.SpectralAnalysis(spectral_choises.index(args.analysis)),
        wav_input_dir = args.wav_input_dir,
        output_dir = args.output_dir if args.output_dir is not None else args.wav_input_dir,
        params = lps_analysis.Parameters(
            n_spectral_pts=args.n_spectral_pts,
            overlap=args.overlap,
            n_mels=args.n_mels,
            decimation_rate=args.decimation_rate,
            log_scale=args.log_scale
        ),
        integration = None if args.integration_interval is None else
                lps_analysis.TimeIntegration(integration_interval=args.integration_interval,
                        integration_overlap=args.integration_overlap,
                        in_seconds=args.in_seconds),
        plot_type=lps_manager.PlotType(plot_choices.index(args.plot_type)),
        frequency_in_x_axis=args.frequency_in_x_axis,
        colormap=plt.get_cmap(args.colormap),
        override=args.override,
        n_files=args.n_files
    )
