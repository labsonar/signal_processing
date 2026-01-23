import os
import argparse

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

import lps_sp.signal as lps_signal
import lps_sp.acoustical.broadband as lps

def main(fs, ref_db, n_samples):
    """Plot the PSDs for all noise types."""

    base_dir = "./result"
    os.makedirs(base_dir, exist_ok = True)

    norm = lps_signal.Normalization.MIN_MAX_ZERO_CENTERED

    plt.figure(figsize=(10, 6))

    theoretical_color_dict = {
        lps.ColoredNoises.WHITE: "gray",
        lps.ColoredNoises.PINK: "lightpink",
        lps.ColoredNoises.BROWN: "sandybrown",
        lps.ColoredNoises.BLUE: "deepskyblue",
        lps.ColoredNoises.VIOLET: "violet"
    }

    estimated_color_dict = {
        lps.ColoredNoises.WHITE: "black",
        lps.ColoredNoises.PINK: "pink",
        lps.ColoredNoises.BROWN: "peru",
        lps.ColoredNoises.BLUE: "royalblue",
        lps.ColoredNoises.VIOLET: "darkviolet"
    }


    for color in lps.ColoredNoises:

        freqs_theoretical, psd_theoretical = color.to_psd(fs=fs, ref_db=ref_db)

        noise, _ = color.generate(n_samples=n_samples, fs=fs, ref_db=ref_db)
        freqs_estimated, psd_estimated = lps.psd(noise, fs=fs)

        plt.plot(freqs_theoretical, psd_theoretical,
                    color=theoretical_color_dict[color], label=f'{color} (Theoretical)')
        plt.plot(freqs_estimated, psd_estimated, '--',
                    color=estimated_color_dict[color], label=f'{color} (Estimated)')

        wav.write(os.path.join(base_dir,f'{color}.wav'), fs, norm(noise))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Power Spectral Density of Colored Noises')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.semilogx()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir,'colored_noises.png'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Generate and save colored noises as .wav files with PSD plots.")

    parser.add_argument("--fs", type=int, default=48000,
                            help="Sampling frequency in Hz (default: 48000).")
    parser.add_argument("--ref_db", type=float, default=50,
                            help="Reference level in dB (default: 50).")
    parser.add_argument("--duration", type=float, default=10,
                            help="Duration of the noise in seconds (default: 10).")

    args = parser.parse_args()

    main(args.fs, args.ref_db, int(args.fs * args.duration))
