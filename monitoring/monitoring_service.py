import uuid
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from drift_detection.detect_drift import DataDriftDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = (
    PROJECT_ROOT / "monitoring" / "reference_data"
    / "reference_dataset.csv"
)

REPORT_PATH = (
    PROJECT_ROOT / "monitoring" / "reports"
    / "drift_report.json"
)

PREDICTION_LOG_PATH = (
    PROJECT_ROOT / "monitoring" / "prediction_logs"
    / "predictions.jsonl"
)

MIN_CURRENT_SAMPLES = 100
CURRENT_WINDOW_SIZE = 500


def load_prediction_features():
    """Load actual prediction inputs from JSONL logs."""

    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()

    records = []

    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                features = entry.get("features", {})

                if features:
                    records.append(features)

            except (json.JSONDecodeError, AttributeError):
                continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records[-CURRENT_WINDOW_SIZE:])


def run_drift_analysis():
    """Compare real prediction inputs against fixed reference data."""

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Reference dataset not found. Run create_reference_data.py first."
        )

    reference_data = pd.read_csv(REFERENCE_PATH)
    current_data = load_prediction_features()

    current_data = current_data.reindex(
        columns=reference_data.columns
    )

    current_data = current_data.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    sample_count = len(current_data)

    if sample_count < MIN_CURRENT_SAMPLES:

        report = {
            "analysis_status": "insufficient_data",
            "analysis_type": "real_prediction_inputs",
            "total_predictions": sample_count,
            "minimum_required": MIN_CURRENT_SAMPLES,
            "drift_detected": None,
            "drift_percentage": None,
            "message": (
                f"Need at least {MIN_CURRENT_SAMPLES} valid predictions. "
                f"Currently have {sample_count}."
            )
        }

    else:

        detector = DataDriftDetector(threshold=0.05)

        report = detector.detect(
            reference_data,
            current_data
        )

        report["analysis_status"] = "completed"
        report["analysis_type"] = "real_prediction_inputs"
        report["total_predictions"] = sample_count
        report["minimum_required"] = MIN_CURRENT_SAMPLES

    report["generated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    report["data_source"] = str(PREDICTION_LOG_PATH)
    report["reference_source"] = str(REFERENCE_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report


def get_latest_report():
    """Return the latest saved drift report."""

    if not REPORT_PATH.exists():
        return {
            "status": "not_available",
            "message": "No drift report found. Run an analysis first."
        }

    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_monitoring_status():
    """Return monitoring status."""

    if not REPORT_PATH.exists():
        return {
            "monitoring_status": "waiting",
            "report_available": False
        }

    report = get_latest_report()

    return {
        "monitoring_status": report.get("analysis_status", "unknown"),
        "report_available": True,
        "drift_detected": report.get("drift_detected"),
        "drift_percentage": report.get("drift_percentage"),
        "total_predictions": report.get("total_predictions", 0),
        "minimum_required": report.get(
            "minimum_required",
            MIN_CURRENT_SAMPLES
        ),
        "generated_at": report.get("generated_at")
    }


def log_prediction(features, prediction, probabilities, model_version):
    """Store each prediction with a unique ID."""

    PREDICTION_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    log_entry = {
        "prediction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": int(prediction),
        "probabilities": {
            key: float(value)
            for key, value in probabilities.items()
        },
        "model_version": str(model_version)
    }

    with open(
        PREDICTION_LOG_PATH,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(json.dumps(log_entry) + "\n")

    return log_entry