import json
import logging
from pathlib import Path

from sklearn.datasets import load_breast_cancer

from drift_detection.detect_drift import DataDriftDetector

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "monitoring" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run_monitoring():

    logging.info("Loading reference dataset")

    dataset = load_breast_cancer(as_frame=True)

    reference_data = dataset.data.copy()

    # Simulate incoming production data
    current_data = reference_data.copy()

    current_data["mean radius"] += 5

    detector = DataDriftDetector(threshold=0.05)

    report = detector.detect(
        reference_data,
        current_data
    )

    report_path = REPORT_DIR / "drift_report.json"

    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)

    print("\nMODEL MONITORING REPORT")
    print("=======================")
    print("Total features:", report["total_features"])
    print("Drifted features:", report["drifted_features"])
    print("Drift percentage:", round(report["drift_percentage"], 2), "%")
    print("Drift detected:", report["drift_detected"])

    print("\nDRIFTED FEATURES")

    for feature in report["feature_results"]:
        if feature["drift_detected"]:
            print(
                feature["feature"],
                "| p-value:",
                round(feature["p_value"], 6)
            )

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    run_monitoring()
    