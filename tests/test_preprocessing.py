import pandas as pd

from ml_pipeline.preprocessing.preprocess import preprocess_dataset


def test_preprocessing():

    data = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "salary": [20000, 25000, 30000, 35000, 40000,
                   45000, 50000, 55000, 60000, 65000],
        "city": ["Mumbai", "Pune", "Mumbai", "Pune", "Mumbai",
                 "Pune", "Mumbai", "Pune", "Mumbai", "Pune"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    result = preprocess_dataset(data, "target")

    assert result["X_train"].shape[0] == 8

    assert result["X_test"].shape[0] == 2

    assert result["X_train"].shape[1] == 4

    assert result["X_test"].shape[1] == 4


def test_missing_values():

    data = pd.DataFrame({
        "age": [20, 25, None, 35, 40, 45, 50, 55, 60, 65],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    result = preprocess_dataset(data, "target")

    assert result["X_train"].shape[0] == 8


def test_invalid_target():

    data = pd.DataFrame({
        "age": [20, 25, 30],
        "target": [0, None, 1]
    })

    try:
        preprocess_dataset(data, "target")
        assert False, "Expected ValueError"

    except ValueError:
        assert True