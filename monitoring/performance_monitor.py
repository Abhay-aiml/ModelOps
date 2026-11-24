import json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "prediction_logs"
    / "predictions.jsonl"
)

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "prediction_logs"
    / "ground_truth.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "reports"
    / "performance_report.json"
)


# ==========================================
# GROUND TRUTH COLLECTION
# ==========================================

def record_ground_truth(prediction_id, actual_label):

    if actual_label not in (0, 1):
        raise ValueError("Actual label must be 0 or 1.")

    if not PREDICTION_LOG_PATH.exists():
        raise ValueError("Prediction log does not exist.")

    prediction_ids = set()

    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entry = json.loads(line)
                if "prediction_id" in entry:
                    prediction_ids.add(entry["prediction_id"])

    if prediction_id not in prediction_ids:
        raise ValueError("Prediction ID not found.")

    GROUND_TRUTH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Prevent duplicate ground-truth submissions
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    existing = json.loads(line)
                    if existing["prediction_id"] == prediction_id:
                        raise ValueError(
                            "Ground truth already recorded for this prediction."
                        )

    record = {
        "prediction_id": prediction_id,
        "actual_label": actual_label,
        "recorded_at": datetime.now(timezone.utc).isoformat()
    }

    with open(GROUND_TRUTH_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


# ==========================================
# LOAD VERIFIED PREDICTIONS AND LABELS
# ==========================================

def load_evaluation_data():

    if not PREDICTION_LOG_PATH.exists():
        return [], [], []

    if not GROUND_TRUTH_PATH.exists():
        return [], [], []

    predictions = {}

    with open(PREDICTION_LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entry = json.loads(line)
                if "prediction_id" in entry:
                    predictions[entry["prediction_id"]] = entry

    y_true = []
    y_pred = []
    model_versions = []

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            truth = json.loads(line)
            prediction_id = truth["prediction_id"]

            if prediction_id not in predictions:
                continue

            y_true.append(int(truth["actual_label"]))

            y_pred.append(
                int(predictions[prediction_id]["prediction"])
            )

            model_versions.append(
                str(predictions[prediction_id]["model_version"])
            )

    return y_true, y_pred, model_versions


# ==========================================
# PERFORMANCE ANALYTICS
# ==========================================

def calculate_performance():

    y_true, y_pred, versions = load_evaluation_data()

    if not y_true:
        return {
            "status": "insufficient_data",
            "evaluated_predictions": 0,
            "message": "No verified ground-truth labels available."
        }

    version_groups = defaultdict(lambda: ([], []))

    for actual, predicted, version in zip(
        y_true, y_pred, versions
    ):
        version_groups[version][0].append(actual)
        version_groups[version][1].append(predicted)

    version_metrics = {}

    for version, (actual, predicted) in version_groups.items():

        version_metrics[version] = {
    "evaluated_predictions": len(actual),
    "accuracy": float(accuracy_score(actual, predicted)),
    "precision": float(precision_score(actual, predicted, pos_label=0, zero_division=0)),
    "recall": float(recall_score(actual, predicted, pos_label=0, zero_division=0)),
    "f1_score": float(f1_score(actual, predicted, pos_label=0, zero_division=0))
}

    report = {
    "status": "completed",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "evaluated_predictions": len(y_true),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    "f1_score": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
    "confusion_matrix": confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).tolist(),
    "model_version_metrics": version_metrics
}

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report