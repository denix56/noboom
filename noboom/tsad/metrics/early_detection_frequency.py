import numpy.typing as npt
import numpy as np

from .utils import _edl

def edf(predictions: npt.NDArray[np.integer], targets: npt.NDArray[np.integer]) -> float:
    """Compute the early detection frequency.

    An anomaly is a maximal continuous segment of targets, where targets is not zero.
    An anomaly is detected early, if the first prediction with value 1 in an anomaly window is in the segment where
    targets are 1.

    Example:
    targets     = [0,1,2,3,0,1,2,3,0]
    predictions = [0,1,1,0,0,0,1,1,0]
    Both anomalies are detected, but only the first is detected early.

    The early detection frequency is the fraction of anomalies that are detected early among all detected anomalies.
    EDF = #early detected anomalies / #detected anomalies

    :param predictions: A binary sequence of predictions.
    :param targets: A sequence of ground-truth labels with elements in [0,1,2,3].
    :return: Early Detection Frequency
    """
    return _edl(predictions, targets, mode="early")
