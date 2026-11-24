import pandas as pd
from sklearn.datasets import load_breast_cancer

from drift_detection.detect_drift import DataDriftDetector


def test_no_drift():

    dataset = load_breast_cancer(as_frame=True)

    reference = dataset.data.copy()
    current = reference.copy()

    detector = DataDriftDetector(threshold=0.05)

    report = detector.detect(reference, current)

    assert report["drift_detected"] is False
    assert report["drift_percentage"] == 0


def test_detect_drift():

    dataset = load_breast_cancer(as_frame=True)

    reference = dataset.data.copy()
    current = reference.copy()

    # Simulate a change in incoming data
    current["mean radius"] = current["mean radius"] + 10

    detector = DataDriftDetector(threshold=0.05)

    report = detector.detect(reference, current)

    assert report["drift_detected"] is True
    assert report["drifted_features"] >= 1


def test_drift_percentage():

    dataset = load_breast_cancer(as_frame=True)

    reference = dataset.data.copy()
    current = reference.copy()

    current["mean radius"] = current["mean radius"] * 3

    detector = DataDriftDetector(threshold=0.05)

    report = detector.detect(reference, current)

    assert report["drift_percentage"] > 0