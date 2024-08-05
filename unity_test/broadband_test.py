"""
This module contains unit tests for the `lps_sp.acoustical.broadband` modules, focusing on
Power Spectral Density (PSD) calculations and broadband noise generation.

Tests:
    test_white_noise: Tests the PSD calculation for white noise, ensuring that the mean PSD 
                      is close to the expected value.
    test_sin: Tests the PSD calculation for a sinusoidal wave, ensuring that the estimated 
              RMS value and background noise level are close to the expected values.
    test_bb: Tests the broadband noise generation and PSD calculation, ensuring that the 
             generated noise spectrum matches the desired spectrum.

Usage:
    This module is intended to be used with a test runner such as unittest. It verifies the
    accuracy and correctness of PSD calculations and noise generation functions from the 
    `lps_sp.acoustical.broadband`.

Example:
    To run the tests, execute the following command:
    
        python -m unittest <name_of_this_file>.py
"""
import unittest
import numpy as np

import lps_sp.acoustical.broadband as lps_bb
import lps_utils.utils as lps_utils

class TestBroadband(unittest.TestCase):
    """Class for unity test in acoustical broadband module."""

    def setUp(self):
        """Set up the test commom configurations."""
        self.fs = 48000
        self.duration = 1
        self.t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)

    def test_white_noise(self):
        """Test PSD calculation for white noise."""
        lps_utils.set_seed()

        desired_psd = 1

        # Generating a white noise (gaussian noise) with psd 1 V**2/Hz
        # Total noise power:
        #    P = ∫ ​psd df  for white noise => P = psd * Δf = psd * fs/2
        # Then calculate the standard deviation for the calculated power
        #   power => P = E[x^2]
        #   standard deviation => std = √(var(x)) = √(E[(x-μ)^2])
        #       as mean (μ) is zero
        #       std = √P
        noise = np.random.normal(0, np.sqrt(desired_psd * self.fs / 2), len(self.t))

        _, psd_values = lps_bb.psd(noise, self.fs, db_unity=False)

        # Check if the PSD mean is close to what is expected
        self.assertAlmostEqual(np.mean(psd_values), desired_psd, delta=desired_psd * 0.05)

    def test_sin(self):
        """Test PSD calculation for sin wave."""
        # based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
        lps_utils.set_seed()

        desired_sin_rms = 10
        desired_bg = 0.001

        sin_amp = desired_sin_rms * np.sqrt(2) #Vrms to Vp
        sin_freq = 4e3

        signal = sin_amp * np.sin(2 * np.pi * sin_freq * self.t)
        signal += np.random.normal(0, np.sqrt(desired_bg * self.fs / 2), len(self.t))

        window_size = 1024
        freqs, psd_values = lps_bb.psd(signal, self.fs, window_size = window_size,
                                       window = 'flattop', db_unity=False)

        estimated_sin_rms = np.sqrt(np.max(psd_values) * (freqs[1] - freqs[0])) * 2

        index = next(i for i, f in enumerate(freqs) if f >= sin_freq)
        background_indexes = np.r_[0:index-5, index+5:psd_values.size]
        estimated_bg = np.mean(psd_values[background_indexes])

        # Check if the PSD mean is close to what is expected
        self.assertAlmostEqual(estimated_sin_rms, desired_sin_rms, delta=desired_sin_rms * 0.05)
        self.assertAlmostEqual(estimated_bg, desired_bg, delta=desired_bg * 0.05)

    def test_bb(self):
        """Test broadband noise generation and PSD calculation."""
        f_test_min = 1000
        f_test_max = 3000
        max_db = 100
        min_db = 60

        n_freqs = 100

        frequencies = np.linspace(0, self.fs / 2, n_freqs)

        # Create a desired spectrum with two levels and a instante transition
        min_sample = next(i for i, f in enumerate(frequencies) if f >= f_test_min)
        max_sample = next(i for i, f in enumerate(frequencies) if f >= f_test_max)
        desired_spectrum = np.ones(len(frequencies)) * min_db
        for i in range(min_sample, max_sample):
            desired_spectrum[i] = max_db

        signal = lps_bb.generate(frequencies, desired_spectrum, len(self.t), self.fs)

        freqs, psd_values = lps_bb.psd(signal=signal, fs=self.fs)


        min_index = next(i for i, f in enumerate(freqs) if f >= f_test_min)
        max_index = next(i for i, f in enumerate(freqs) if f >= f_test_max)

        background_indexes = np.r_[0:min_index-1, max_index+1:psd_values.size]
        min_power = np.mean(psd_values[background_indexes])
        max_power = np.mean(psd_values[min_index+1:max_index-1])

        # Check if the PSD mean is close to what is expected
        self.assertAlmostEqual(max_power, max_db, delta=max_db * 0.05)
        self.assertAlmostEqual(min_power, min_db, delta=min_db * 0.05)

if __name__ == '__main__':
    unittest.main()
