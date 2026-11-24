import logging

from mlflow import MlflowClient


MODEL_NAME = "ModelOpsClassifier"
MINIMUM_F1_SCORE = 0.95

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = MlflowClient(
    tracking_uri="sqlite:///mlflow.db"
)


def promote_to_staging():

    model = client.get_registered_model(MODEL_NAME)

    if not model.latest_versions:
        raise ValueError("No registered model versions found")

    latest = max(
        model.latest_versions,
        key=lambda version: int(version.version)
    )

    if latest.status != "READY":
        raise ValueError("Model version is not READY")

    run = client.get_run(latest.run_id)

    f1_score = run.data.metrics.get("f1_score", 0)

    logger.info(f"Model version: {latest.version}")
    logger.info(f"F1 score: {f1_score}")

    if f1_score < MINIMUM_F1_SCORE:
        raise ValueError("Model failed quality checks")

    client.set_registered_model_alias(
        MODEL_NAME,
        "staging",
        latest.version
    )

    client.set_model_version_tag(
        MODEL_NAME,
        latest.version,
        "validation_status",
        "passed"
    )

    logger.info(
        f"Version {latest.version} promoted to STAGING"
    )

    return latest.version


if __name__ == "__main__":

    version = promote_to_staging()

    print(f"\nSTAGING VERSION: {version}")