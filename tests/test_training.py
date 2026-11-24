from ml_pipeline.training.train import train_models


def test_model_training():

    result = train_models()

    assert result["best_model"] is not None

    assert len(result["results"]) == 3

    for model in result["results"]:

        assert "accuracy" in model["metrics"]
        assert "precision" in model["metrics"]
        assert "recall" in model["metrics"]
        assert "f1_score" in model["metrics"]