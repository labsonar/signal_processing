import os
import math
import argparse
import typing
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib as tikz

import scipy.signal as scipy
import scipy.io.wavfile as wavfile
import librosa
import sympy

import lps_utils.utils as lps_utils

def _get_demon_steps(fs_in, fs_out=50):
    if (fs_in % fs_out) != 0:
        raise ValueError("fs_in não divisível por fs_out")

    factors = sympy.factorint(int(fs_in / fs_out))
    factor_list = [f for factor, count in factors.items() for f in [factor] * count]

    decimate_ratio1 = 1
    decimate_ratio2 = 1
    add_one = True

    while factor_list:
        if len(factor_list) == 1:
            part1 = factor_list.pop()
            part2 = 1
        elif len(factor_list) == 2:
            part1 = factor_list.pop(0)
            part2 = factor_list.pop()
        else:
            part1 = factor_list.pop(0) * factor_list.pop()
            part2 = 1

        if add_one:
            decimate_ratio1 *= part1
            decimate_ratio2 *= part2
        else:
            decimate_ratio1 *= part2
            decimate_ratio2 *= part1
        add_one = not add_one

    return [decimate_ratio1, decimate_ratio2]

def _demon(data, fs, n_fft=512, max_freq=50, overlap_ratio=0.25,
           apply_bandpass=True, bandpass_specs=None, method='abs'):

    [decimate_ratio1, decimate_ratio2] = _get_demon_steps(fs, max_freq)
    x = data.copy()

    nyq = fs / 2
    if apply_bandpass:
        if bandpass_specs is None:
            wp = [1000 / nyq, 2000 / nyq]
            ws = [700 / nyq, 2300 / nyq]
            rp = 0.5
            As = 50
        else:
            fp = bandpass_specs["fp"]
            fs_band = bandpass_specs["fs"]
            wp = np.array(fp) / nyq
            ws = np.array(fs_band) / nyq
            rp = bandpass_specs["rs"]
            As = bandpass_specs["as"]

        N, wc = scipy.cheb2ord(wp, ws, rp, As)
        b, a = scipy.cheby2(N, rs=As, Wn=wc, btype='bandpass', output='ba', analog=True)
        x = scipy.lfilter(b, a, x, axis=0)

    if method == 'hilbert':
        x = scipy.hilbert(x)
    elif method == 'abs':
        x = np.abs(x)
    else:
        raise ValueError("Método inválido")

    x = scipy.decimate(x, decimate_ratio1, ftype='fir', zero_phase=False)
    x = scipy.decimate(x, decimate_ratio2, ftype='fir', zero_phase=False)

    final_fs = (fs // decimate_ratio1) // decimate_ratio2
    x = x / np.max(np.abs(x))
    x = x - np.mean(x)

    fft_over = math.floor(n_fft - 2 * max_freq * overlap_ratio)
    sxx = librosa.stft(x, window='hann', win_length=n_fft,
                       hop_length=n_fft - fft_over, n_fft=n_fft)
    freq = librosa.fft_frequencies(sr=final_fs, n_fft=n_fft)
    time = librosa.frames_to_time(np.arange(sxx.shape[1]), sr=final_fs,
                                  hop_length=(n_fft - fft_over))

    sxx = np.abs(sxx)
    sxx, freq = sxx[8:, :], freq[8:]

    return np.transpose(sxx), freq, time

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

    parser = argparse.ArgumentParser(description="Run DEMON analysis on one or more .wav files.")
    parser.add_argument("input", type=str, help="WAV file, comma-separated list, or directory")
    parser.add_argument("--n_fft", type=int, default=512, help="FFT size")
    parser.add_argument("--max_freq", type=int, default=50, help="Max freq for DEMON (Hz)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap ratio [0-1]")
    parser.add_argument("--merge", action="store_true", help="Plot all files in one graph")
    parser.add_argument("--export_tex", action="store_true",
                        help="Export figure to .tex")
    args = parser.parse_args()

    files = _resolve_input_files(args.input)
    files = sorted(files)

    curves = []

    max_list = []
    for filename in tqdm.tqdm(files, desc="Processing files", ncols=120):
        fs, signal = wavfile.read(filename)
        if signal.ndim > 1:
            signal = signal[:, 0]
        intensity, freqs, _ = _demon(signal, fs,
                            n_fft=args.n_fft,
                            max_freq=args.max_freq,
                            overlap_ratio=args.overlap)
        avg = np.mean(intensity, axis=0)
        label = os.path.splitext(os.path.basename(filename))[0]
        curves.append((filename, label, freqs * 60, avg))

        max_list.append(np.max(avg))


    if args.merge:
        plt.figure()
        offset_step = 1
        for i, (_, label, freqs, avg) in enumerate(curves):
            norm_avg = avg / np.max(max_list)
            offset = (len(curves) - i - 1) * offset_step
            plt.plot(freqs, norm_avg + offset, label=label)
        plt.xlabel("Frequency [RPM]")
        plt.ylabel("Intensity")
        plt.yticks([])
        plt.legend(loc="upper right")
        plt.tight_layout()

        out_dir = os.path.dirname(files[0])
        out_base = os.path.join(out_dir, "merged_demon")
        if args.export_tex:
            tikz.save(out_base + ".tex")
        else:
            plt.savefig(out_base + ".png")
        plt.close()

    else:
        for filename, label, freqs, avg in curves:
            plt.figure()
            plt.plot(freqs, avg)
            plt.title("DEMON")
            plt.xlabel("Frequency [RPM]")
            plt.ylabel("Intensity")
            plt.tight_layout()

            out_dir = os.path.dirname(filename)
            out_name = os.path.splitext(os.path.basename(filename))[0] + "_demon"
            out_path = os.path.join(out_dir, out_name)

            if args.export_tex:
                tikz.save(out_path + ".tex")
            else:
                plt.savefig(out_path + ".png")
            plt.close()

if __name__ == "__main__":
    main()
