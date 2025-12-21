import numpy as np

from noboom.tsad.metrics import edf, ldf


def test_edf_and_ldf_examples_match_documentation():
    targets = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])
    predictions = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0])

    # First anomaly detected early (index 1 < first target >1), second not.
    assert edf(predictions, targets) == 0.5

    # No anomaly has its first detection in the targets==3 region.
    assert ldf(predictions, targets) == 0.0


def test_edf_and_ldf_bounds_and_behavior():
    targets = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])

    # All detections happen immediately -> fully early, never late.
    early_predictions = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0])
    assert edf(early_predictions, targets) == 1.0
    assert ldf(early_predictions, targets) == 0.0

    # Detections only in the targets == 3 region -> never early, always late.
    late_predictions = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0])
    assert edf(late_predictions, targets) == 0.0
    assert ldf(late_predictions, targets) == 1.0

    # Both metrics are frequencies and must stay within [0, 1].
    assert 0.0 <= edf(late_predictions, targets) <= 1.0
    assert 0.0 <= ldf(late_predictions, targets) <= 1.0


def test_detection_is_best_when_predictions_match_targets():
    targets = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])

    # Perfect predictions raise EDF and suppress LDF.
    perfect = targets
    delayed = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0])

    perfect_edf = edf(perfect, targets)
    perfect_ldf = ldf(perfect, targets)

    delayed_edf = edf(delayed, targets)
    delayed_ldf = ldf(delayed, targets)

    assert perfect_edf > delayed_edf
    assert perfect_ldf < delayed_ldf


def test_multiple_events_only_first_detection_counts():
    targets = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0])

    # First anomaly: prediction only in the early region.
    # Second anomaly: first prediction arrives late, even though an earlier
    # false alarm exists outside the event window.
    predictions = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0])

    # Early for the first anomaly, late for the second -> split evenly.
    assert edf(predictions, targets) == 0.5
    assert ldf(predictions, targets) == 0.5
