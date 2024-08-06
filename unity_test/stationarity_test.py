"""
This module contains unit tests for the `lps_sp.stationarity` module.

Tests:
    test_dissimilarit_measures: Test consistence in all stationarity statistical test.

Example:
    To run the tests, execute the following command:
    
        python -m unittest stationarity_test.py
"""
import unittest
import numpy as np

import lps_utils.utils as lps_utils
import lps_sp.stationarity as lps

class TestStatisticalTests(unittest.TestCase):
    """ Unit tests for the statistical tests of stationarity. """

    def setUp(self):
        """Set up the test commom configurations."""
        lps_utils.set_seed()

        self.n_samples = 10000
        self.fs = 48e3
        self.t = np.arange(self.n_samples) / self.fs
        frequency = 5e3

        self.stationary_data = np.sin(2 * np.pi * frequency * self.t) \
                    + np.random.normal(0, 0.05, self.n_samples)

        self.non_stationary_data = self.stationary_data + np.arange(self.n_samples)


    def test_dissimilarit_measures(self):
        """ Test consistence in all stationarity statistical test. """

        for test in lps.StatisticalTest:
            self.assertTrue(test.is_stationary(self.stationary_data))
            self.assertFalse(test.is_stationary(self.non_stationary_data))


if __name__ == '__main__':
    unittest.main()
