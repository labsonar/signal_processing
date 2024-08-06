"""
This module provides functionality to handle anomalies in a time series.
"""
import typing
import numpy as np

class ZScoreDetector:
    """ Class to represent a z-score anomaly detector. """

    def __init__(self, estimation_window_size: int, normalization_step: int = 1):
        """
        Parameters:
        - estimation_window (int): The size of the moving window to calculate mean and std
            deviation.
        - normalization_step (int, optional): The step size between samples for detection.
            Default to 1.
        """
        self.estimation_window_size = estimation_window_size
        self.normalization_step = normalization_step

    def detect(self, input_data: np.array, threshold: float = 3) -> np.array:
        """
        Perform Z-score based anomaly detector on the given data.

        Parameters:
        - input_data (np.array): The input data array for detection.
        - threshold (float, optional): The Z-score threshold to detect anomalies (default is 3.0).

        Returns:
        - np.array: An array of indices (center of the analysis window) where anomalies are
            detected.
        """
        anomalies = []
        z_scoress = []
        for i in range(self.estimation_window_size + self.normalization_step,
                       len(input_data),
                       self.normalization_step):

            start_index = i-self.estimation_window_size-self.normalization_step

            estimation_window = input_data[start_index:start_index+self.estimation_window_size]
            mean = np.mean(estimation_window)
            std = np.std(estimation_window)

            sample_window = input_data[i-self.normalization_step:i]
            z_scores = np.abs((np.median(sample_window) - mean) / std)

            if z_scores > threshold:
                anomalies.append(i - self.normalization_step)

            z_scoress.append(np.median(sample_window))

        anomalies = np.array(anomalies)

        if len(anomalies) > 1:
            diffs = np.diff(anomalies)
            to_keep = np.insert(diffs > self.normalization_step, 0, True)
            anomalies = anomalies[to_keep]

        return anomalies

    def evaluate(self,
                 input_data: np.array,
                 expected_detections: typing.List[int],
                 threshold: float = 3,
                 tolerance: int = -1) -> typing.Tuple[float, float]:
        """
        Perform Z-score based anomaly detector and evalute the detection in reference to the
            expected detections on the given data.

        Parameters:
        - input_data (np.array): The input data array for detection.
        - expected_detections (typing.List[int]): List of indexes where detection are expected.
        - threshold (float): The Z-score threshold to detect anomalies (default is 3.0).
        - tolerance (int): The number of samples (default is window_size//2).

        Returns:
        - typing.Tuple[float, float]: A tuple with TP and FP:
            TP (True positive) fractions of corrected identified detections.
            FP (false negative) fractions of analised window outside expected detections that
                generates a wrong detections.
        """
        anomalies = self.detect(input_data, threshold)

        tp = 0
        fp = 0

        if tolerance <= 0:
            tolerance = self.estimation_window_size//2

        for index in expected_detections:
            is_detected = np.any((anomalies >= index - tolerance) &
                                 (anomalies <= index + tolerance))
            if is_detected:
                tp += 1

        negative_samples = len(input_data) - self.estimation_window_size * len(expected_detections)
        negative_n_windows = (negative_samples - self.estimation_window_size)\
                //self.normalization_step + 1
        fp = (len(anomalies) - tp)/negative_n_windows
        tp = tp/len(expected_detections)

        return tp, fp
