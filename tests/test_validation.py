import pandas as pd

from ml_pipeline.data_validation.validate import validate_dataset


def test_valid_dataset():

    data = pd.DataFrame({
        "age": [20, 25, 30],
        "salary": [20000, 30000, 40000],
        "target": [0, 1, 0]
    })

    result = validate_dataset(data, "target")

    assert result["valid"] is True


def test_missing_target():

    data = pd.DataFrame({
        "age": [20, 25, 30]
    })

    result = validate_dataset(data, "target")

    assert result["valid"] is False


def test_empty_dataset():

    data = pd.DataFrame()

    result = validate_dataset(data, "target")

    assert result["valid"] is False