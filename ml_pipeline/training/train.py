import logging
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from ml_pipeline.preprocessing.preprocess import preprocess_dataset


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def train_models():

    logger.info("Loading dataset")

    dataset = load_breast_cancer(as_frame=True)
    data = dataset.frame

    result = preprocess_dataset(
        data,
        target_column="target"
    )

    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    preprocessor = result["preprocessor"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ModelOps-Experiments")

    results = []

    for model_name, model in models.items():

        logger.info(f"Training {model_name}")

        with mlflow.start_run(run_name=model_name):

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, zero_division=0
                )
            }

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)

            serving_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])
            artifact_bundle = {
                "preprocessor": preprocessor,
                "model": model,
                "feature_names": list(dataset.feature_names)
                
            }

            mlflow.sklearn.log_model(
                serving_pipeline,
                name="model",
                serialization_format="pickle"
                
            )

            results.append({
                "name": model_name,
                "model": model,
                "metrics": metrics,
                "pipeline": artifact_bundle,
                "run_id": mlflow.active_run().info.run_id
            })

            logger.info(f"{model_name}: {metrics}")

    best_result = max(
        results,
        key=lambda result: result["metrics"]["f1_score"]
    )
    # Register the winning model in MLflow

    model_uri = f"runs:/{best_result['run_id']}/model"

    registered_version = mlflow.register_model(
       model_uri=model_uri,
       name="ModelOpsClassifier"
    )

    logger.info(
        f"Registered model: {registered_version.name}"
    )

    logger.info(
        f"Model version: {registered_version.version}"
    )

    joblib.dump(
        best_result["pipeline"],
        ARTIFACT_DIR / "best_model.pkl"
    )

    logger.info(f"Best model: {best_result['name']}")

    return {
        "best_model": best_result["name"],
        "registered_model": registered_version.name,
        "registered_version": registered_version.version,
        "results": [
            {
                "name": result["name"],
                "metrics": result["metrics"]
            }
            for result in results
        ]
    }


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    output = train_models()

    print("\nMODEL COMPARISON")
    print("================")

    for result in output["results"]:
        print(result)

    print("\nBEST MODEL:", output["best_model"])