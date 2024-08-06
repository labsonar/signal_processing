"""
This module provides a utility for performing statistical tests to check the stationarity of time
series data.
"""
import enum
import numpy as np

import arch.unitroot as arch_root

class StatisticalTest(enum.Enum):
    """Enum class to represent statistical tests of stationarity."""
    ADF = 0
    AUGMENTEDDICKEYFULLER = 0
    KPSS = 1
    KWIATKOWSKIPHILLIPSSCHMIDTSHIN = 1
    PP = 2
    PHILLIPSPERRON = 2

    def __str__(self):
        """Returns the string representation of the statistical test."""
        return str(self.name).rsplit('.', maxsplit=1)[-1].upper()

    def get_info(self, data: np.array):
        """
        Runs the selected statistical test on the provided data.

        Args:
            data (np.ndarray): The data on which to perform the test.

        Returns:
            The result of the statistical test.
        """
        if self == StatisticalTest.ADF:
            result = arch_root.ADF(data)

        elif self == StatisticalTest.KPSS:
            result = arch_root.KPSS(data, lags=-1)

        elif self == StatisticalTest.PP:
            result = arch_root.PhillipsPerron(data)

        else:
            raise NotImplementedError(f"StatisticalTest not implemented for {self}")

        return result

    def calc_state(self, data: np.array) -> float:
        """
        Returns the test statistic from the selected statistical test.

        Args:
            data (np.ndarray): The data on which to perform the test.

        Returns:
            float: The value of test statistic.
        """
        return self.get_info(data).stat

    def is_stationary(self, data: np.array, pvalue: float = 0.05) -> bool:
        """
        Determines if the provided data is stationary based for this statistical test.

        Args:
            data (np.ndarray): The data on which to perform the test.
            pvalue (float, optional): The significance level to determine stationarity.
                Defaults to 0.05.

        Returns:
            bool: True if the series is stationary, False otherwise.
        """
        if self == StatisticalTest.KPSS:
            return self.get_info(data).pvalue >= pvalue
        return self.get_info(data).pvalue < pvalue
