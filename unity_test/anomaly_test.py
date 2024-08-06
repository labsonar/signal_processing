"""
This module contains unit tests for the `lps_sp.anomaly` modules.

Tests:
    test_detect: Test detect method from ZScoreDetector.
    test_evaluate: Test evaluate method from ZScoreDetector.

Example:
    To run the tests, execute the following command:
    
        python -m unittest anomaly_test.py
"""
import unittest
import numpy as np

import lps_utils.utils as lps_utils
import lps_sp.anomaly as lps

class TestZScoreDetector(unittest.TestCase):
    """Class for unity test in anomaly module."""

    def setUp(self):
        """Set up the test commom configurations."""
        lps_utils.set_seed()

        self.n_samples = int(1e4)
        self.window_size = 500
        self.sample_step = 5
        self.expected_anomalies = 1050 # Introduce an anomaly
        self.detector = lps.ZScoreDetector(self.window_size, self.sample_step)

        self.data_stationary = np.random.normal(0, 1, self.n_samples)

        self.data_non_stationary = self.data_stationary.copy()
        start = self.expected_anomalies
        end = self.expected_anomalies + 10 * self.sample_step
        self.data_non_stationary[start:end] = 5


    def test_detect(self):
        """Test detect method."""

        # Test detection in stationary data
        anomalies = self.detector.detect(self.data_stationary)
        self.assertEqual(len(anomalies), 0, "Should not detect anomalies in stationary data")

        # Test detection in non-stationary data
        anomalies = self.detector.detect(self.data_non_stationary)
        self.assertTrue(len(anomalies) == 1, "Should detect the inserted anomaly")
        self.assertTrue(any(np.abs(anomalies - self.expected_anomalies) < self.sample_step),
                        "Should detect anomaly near index 450")

    def test_evaluate(self):
        """Test evaluate method."""

        tp, fp = self.detector.evaluate(self.data_non_stationary, [self.expected_anomalies])
        self.assertAlmostEqual(tp, 1.0, delta=0.01,
            msg="True Positive Rate should be close to 1.0")
        self.assertAlmostEqual(fp, 0.0, delta=0.01,
            msg="False Positive Rate should be close to 0.0")


if __name__ == '__main__':
    unittest.main()
