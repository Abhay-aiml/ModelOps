import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "prediction_logs"
    / "predictions.jsonl"
)


def load_predictions():

    if not LOG_PATH.exists():
        return []

    predictions = []

    with open(LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                predictions.append(json.loads(line))

    return predictions


def get_prediction_analytics():

    predictions = load_predictions()

    if not predictions:
        return {
            "total_predictions": 0,
            "message": "No predictions recorded yet."
        }

    labels = [
        "malignant" if item["prediction"] == 0 else "benign"
        for item in predictions
    ]

    prediction_counts = Counter(labels)

    versions = Counter(
        str(item["model_version"])
        for item in predictions
    )

    confidences = [
        max(item["probabilities"].values())
        for item in predictions
    ]

    average_confidence = sum(confidences) / len(confidences)

    return {
        "total_predictions": len(predictions),
        "prediction_distribution": dict(prediction_counts),
        "model_version_distribution": dict(versions),
        "average_confidence": round(average_confidence, 6)
    }