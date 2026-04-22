import os
import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig


class AudioDebugger:
    _instance = None

    def __init__(self):
        self.data = {}

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = AudioDebugger()
        return cls._instance

    @classmethod
    def register(cls, signal_id: str, signal: np.ndarray, fs: lps_qty.Frequency):
        debugger = cls.get()
        debugger.data[signal_id] = (signal, fs)

    @classmethod
    def save(cls, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        debugger = cls.get()

        for signal_id, (signal, fs) in debugger.data.items():
            lps_sig.save_wav(signal, fs, os.path.join(output_dir, f"{signal_id}.wav"))

        debugger.data.clear()
