from pathlib import Path
from sklearn.datasets import load_breast_cancer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "reference_data"
    / "reference_dataset.csv"
)


def create_reference_dataset():

    dataset = load_breast_cancer(as_frame=True)

    reference_data = dataset.data.copy()

    REFERENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    reference_data.to_csv(
        REFERENCE_PATH,
        index=False
    )

    print("Reference dataset created successfully!")
    print("Total rows:", len(reference_data))
    print("Total features:", len(reference_data.columns))
    print("Saved at:", REFERENCE_PATH)


if __name__ == "__main__":
    create_reference_dataset()