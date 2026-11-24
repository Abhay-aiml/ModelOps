import json
import pytest

from monitoring import performance_monitor as pm


@pytest.fixture
def setup_logs(tmp_path, monkeypatch):
    prediction_path = tmp_path / "predictions.jsonl"
    truth_path = tmp_path / "ground_truth.jsonl"
    report_path = tmp_path / "performance_report.json"

    monkeypatch.setattr(pm, "PREDICTION_LOG_PATH", prediction_path)
    monkeypatch.setattr(pm, "GROUND_TRUTH_PATH", truth_path)
    monkeypatch.setattr(pm, "REPORT_PATH", report_path)

    predictions = [
        {
            "prediction_id": "test-001",
            "prediction": 0,
            "model_version": "4"
        },
        {
            "prediction_id": "test-002",
            "prediction": 1,
            "model_version": "4"
        }
    ]

    with open(prediction_path, "w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + "\n")

    return prediction_path, truth_path, report_path


def test_record_ground_truth(setup_logs):
    record = pm.record_ground_truth("test-001", 0)

    assert record["prediction_id"] == "test-001"
    assert record["actual_label"] == 0


def test_duplicate_ground_truth(setup_logs):
    pm.record_ground_truth("test-001", 0)

    with pytest.raises(ValueError, match="already recorded"):
        pm.record_ground_truth("test-001", 0)


def test_invalid_ground_truth_label(setup_logs):
    with pytest.raises(ValueError):
        pm.record_ground_truth("test-001", 2)


def test_unknown_prediction_id(setup_logs):
    with pytest.raises(ValueError, match="Prediction ID not found"):
        pm.record_ground_truth("unknown-id", 0)


def test_performance_calculation(setup_logs):
    pm.record_ground_truth("test-001", 0)
    pm.record_ground_truth("test-002", 1)

    report = pm.calculate_performance()

    assert report["status"] == "completed"
    assert report["evaluated_predictions"] == 2
    assert report["accuracy"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1_score"] == 1.0
    assert "4" in report["model_version_metrics"]


def test_empty_evaluation_data(setup_logs):
    report = pm.calculate_performance()

    assert report["status"] == "insufficient_data"
    assert report["evaluated_predictions"] == 0