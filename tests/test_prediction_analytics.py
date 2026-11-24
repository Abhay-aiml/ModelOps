import json

from monitoring import prediction_analytics


def test_load_predictions(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"

    records = [
        {
            "prediction": 0,
            "probabilities": {"malignant": 0.9, "benign": 0.1},
            "model_version": "4"
        },
        {
            "prediction": 1,
            "probabilities": {"malignant": 0.2, "benign": 0.8},
            "model_version": "4"
        }
    ]

    with open(log_file, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    monkeypatch.setattr(prediction_analytics, "LOG_PATH", log_file)

    predictions = prediction_analytics.load_predictions()

    assert len(predictions) == 2


def test_prediction_analytics(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"

    records = [
        {
            "prediction": 0,
            "probabilities": {"malignant": 0.9, "benign": 0.1},
            "model_version": "4"
        },
        {
            "prediction": 1,
            "probabilities": {"malignant": 0.2, "benign": 0.8},
            "model_version": "4"
        },
        {
            "prediction": 0,
            "probabilities": {"malignant": 0.7, "benign": 0.3},
            "model_version": "5"
        }
    ]

    with open(log_file, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    monkeypatch.setattr(prediction_analytics, "LOG_PATH", log_file)

    result = prediction_analytics.get_prediction_analytics()

    assert result["total_predictions"] == 3
    assert result["prediction_distribution"]["malignant"] == 2
    assert result["prediction_distribution"]["benign"] == 1
    assert result["model_version_distribution"]["4"] == 2
    assert result["model_version_distribution"]["5"] == 1
    assert result["average_confidence"] == 0.8


def test_empty_prediction_logs(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"
    log_file.write_text("", encoding="utf-8")

    monkeypatch.setattr(prediction_analytics, "LOG_PATH", log_file)

    result = prediction_analytics.get_prediction_analytics()

    assert result["total_predictions"] == 0
    assert result["message"] == "No predictions recorded yet."


def test_missing_prediction_log(tmp_path, monkeypatch):
    log_file = tmp_path / "missing.jsonl"

    monkeypatch.setattr(prediction_analytics, "LOG_PATH", log_file)

    predictions = prediction_analytics.load_predictions()

    assert predictions == []