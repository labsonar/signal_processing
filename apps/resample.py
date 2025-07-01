import argparse
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import os
import typing

import lps_sp.signal as lps_signal


def resample_up(data: np.ndarray, orig_fs: int, new_fs: int) -> np.ndarray:
    """
    Resample using non-circular polyphase filtering (avoids FFT artifacts).
    """
    gcd = np.gcd(orig_fs, new_fs)
    up = new_fs // gcd
    down = orig_fs // gcd
    return scipy.signal.resample_poly(data, up=up, down=down)


def build_output_filename(input_file: str, new_fs: int) -> str:
    base, ext = os.path.splitext(input_file)
    return f"{base}_{new_fs}{ext}"


def process_channel(channel_data: np.ndarray, orig_fs: int, new_fs: int) -> np.ndarray:
    if new_fs < orig_fs:
        rate = orig_fs / new_fs
        return lps_signal.decimate(channel_data.astype(np.float64), rate)
    elif new_fs > orig_fs:
        return resample_up(channel_data.astype(np.float64), orig_fs, new_fs)
    else:
        return channel_data.astype(np.float64)


def main(new_fs: float, input_file: str, output_file: str):

    output_file = output_file or build_output_filename(input_file, new_fs)

    orig_fs, data = wavfile.read(input_file)

    print(f"Original sampling rate: {orig_fs} Hz")
    print(f"Target sampling rate: {new_fs} Hz")
    print(f"Number of channels: {data.shape[1] if data.ndim > 1 else 1}")
    print(f"Output file: {output_file}")

    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_channels = data.shape[1]

    resampled_channels = []
    for i in range(n_channels):
        channel = data[:, i]
        resampled = process_channel(channel, orig_fs, new_fs)
        resampled_channels.append(resampled)

    min_len = min(len(ch) for ch in resampled_channels)
    resampled_channels = [ch[:min_len] for ch in resampled_channels]

    output_data = np.stack(resampled_channels, axis=-1)
    lps_signal.save_wav(output_data, new_fs, output_file)

    print(f"Written to {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Resample a WAV file to a new sampling frequency.")
    parser.add_argument("input_file", type=str, help="Path to input WAV file.")
    parser.add_argument("new_fs", type=int, help="New sampling frequency (in Hz).")
    parser.add_argument(
        "--output_file", "-o", type=str, default=None,
        help="Path to output WAV file. If omitted, generates <input>_<new_fs>.wav"
    )

    args = parser.parse_args()

    main(args.new_fs, args.input_file, args.output_file)
