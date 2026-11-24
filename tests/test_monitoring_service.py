import json

import pandas as pd

from monitoring import monitoring_service


def create_test_data(tmp_path, monkeypatch, count):
    reference_path = tmp_path / "reference.csv"
    log_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "drift_report.json"

    reference = pd.DataFrame({
        f"feature_{i}": range(200)
        for i in range(3)
    })

    reference.to_csv(reference_path, index=False)

    with open(log_path, "w", encoding="utf-8") as file:
        for row in range(count):
            entry = {
                "features": {
                    f"feature_{i}": row
                    for i in range(3)
                },
                "prediction": 0,
                "probabilities": {
                    "malignant": 0.9,
                    "benign": 0.1
                },
                "model_version": "4"
            }

            file.write(json.dumps(entry) + "\n")

    monkeypatch.setattr(
        monitoring_service, "REFERENCE_PATH", reference_path
    )
    monkeypatch.setattr(
        monitoring_service, "PREDICTION_LOG_PATH", log_path
    )
    monkeypatch.setattr(
        monitoring_service, "REPORT_PATH", report_path
    )

    return report_path


def test_insufficient_prediction_data(tmp_path, monkeypatch):
    create_test_data(tmp_path, monkeypatch, 33)

    report = monitoring_service.run_drift_analysis()

    assert report["analysis_status"] == "insufficient_data"
    assert report["total_predictions"] == 33
    assert report["minimum_required"] == 100
    assert report["drift_detected"] is None


def test_completed_drift_analysis(tmp_path, monkeypatch):
    create_test_data(tmp_path, monkeypatch, 120)

    report = monitoring_service.run_drift_analysis()

    assert report["analysis_status"] == "completed"
    assert report["total_predictions"] == 120
    assert report["total_features"] == 3
    assert "overall_severity" in report


def test_report_saved_successfully(tmp_path, monkeypatch):
    report_path = create_test_data(
        tmp_path, monkeypatch, 33
    )

    monitoring_service.run_drift_analysis()

    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report["total_predictions"] == 33


def test_monitoring_status(tmp_path, monkeypatch):
    report_path = create_test_data(
        tmp_path, monkeypatch, 33
    )

    monitoring_service.run_drift_analysis()

    status = monitoring_service.get_monitoring_status()

    assert status["report_available"] is True
    assert status["monitoring_status"] == "insufficient_data"
    assert status["total_predictions"] == 33