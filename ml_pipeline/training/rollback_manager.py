import logging
from mlflow import MlflowClient

MODEL_NAME = "ModelOpsClassifier"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = MlflowClient(
    tracking_uri="sqlite:///mlflow.db"
)


def rollback_production(target_version):

    target_version = str(target_version)

    # Verify target version exists
    model = client.get_model_version(
        MODEL_NAME,
        target_version
    )

    if model.status != "READY":
        raise ValueError("Target model is not READY")

    # Get current production version
    try:
        current = client.get_model_version_by_alias(
            MODEL_NAME,
            "production"
        )

        print("Current production:", current.version)

    except Exception:
        raise ValueError("No production model currently exists")

    if current.version == target_version:
        raise ValueError("Target is already in production")

    # Get target model metrics
    run = client.get_run(model.run_id)

    f1_score = run.data.metrics.get("f1_score", 0)

    print("\nROLLBACK REVIEW")
    print("================")
    print("Current version:", current.version)
    print("Target version:", target_version)
    print("Target F1 score:", round(f1_score, 4))

    approval = input(
        "\nType ROLLBACK to confirm: "
    )

    if approval != "ROLLBACK":
        print("Rollback cancelled")
        return

    # Change production alias
    client.set_registered_model_alias(
        MODEL_NAME,
        "production",
        target_version
    )

    print("\nROLLBACK SUCCESSFUL")
    print("Production version:", target_version)


if __name__ == "__main__":

    version = input("Enter rollback target version: ")

    rollback_production(version)