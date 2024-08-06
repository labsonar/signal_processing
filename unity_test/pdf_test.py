"""
This module contains unit tests for the `lps_sp.pdf` modules.

Tests:
    test_estimate_pdf: Test PDF estimation for two normal distributions.
    test_dissimilarities: Test all dissimilarity measures.

Example:
    To run the tests, execute the following command:
    
        python -m unittest pdf_test.py
"""
import unittest
import numpy as np

import lps_utils.utils as lps_utils
import lps_sp.pdf as lps_pdf

class TestPDFEstimation(unittest.TestCase):
    """ Unit tests for the estimate_pdf function. """

    def setUp(self):
        """Set up the test commom configurations."""
        lps_utils.set_seed()

    def test_estimate_pdf(self):
        """ Test PDF estimation for two normal distributions. """
        # Generate two normal distributions
        data1 = np.random.normal(0, 1, 1000)
        data2 = np.random.normal(0, 1, 1000)

        x1_dist, x2_dist, edges = lps_pdf.estimate_pdf(data1, data2, n_bins=50)

        # Check that the histograms are normalized to form a PDF
        self.assertAlmostEqual(np.sum(x1_dist * np.diff(edges)), 1, places=1)
        self.assertAlmostEqual(np.sum(x2_dist * np.diff(edges)), 1, places=1)
        # Check the siilarity results between two equal distributions
        self.assertAlmostEqual(np.mean(x1_dist - x2_dist), 0, delta=1e-3)

class TestSimilarityMeasures(unittest.TestCase):
    """ Unit tests for the DissimilarityMeasure enum class. """

    def setUp(self):
        """Set up the test commom configurations."""
        lps_utils.set_seed()
        # Generate test normal distributions
        self.data1 = np.random.normal(0, 1, int(1e6))
        self.data2 = np.random.normal(0, 1, int(1e6))
        self.data3 = np.random.normal(1, 2, int(1e6))
        self.n_bins = 50

    def test_dissimilaties(self):
        """ Test all dissimilarity measures. """
        # Test DissimilarityMeasure

        for measure in lps_pdf.DissimilarityMeasure:
            # distance between equals distribuitions near to zero
            diss1 = measure.from_data(self.data1, self.data2)
            self.assertAlmostEqual(diss1, 0, delta=0.01)

            # distance between different distribuitions must be greater than equals distribuitions
            diss2 = measure.from_data(self.data1, self.data3)
            self.assertGreater(diss2, diss1)


if __name__ == '__main__':
    unittest.main()
