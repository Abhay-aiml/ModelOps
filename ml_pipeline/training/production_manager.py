import logging
from mlflow import MlflowClient

MODEL_NAME = "ModelOpsClassifier"
MINIMUM_F1_SCORE = 0.95

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = MlflowClient(
    tracking_uri="sqlite:///mlflow.db"
)


def promote_to_production():

    # Get staging model
    staging = client.get_model_version_by_alias(
        MODEL_NAME,
        "staging"
    )

    run = client.get_run(staging.run_id)

    f1_score = run.data.metrics.get("f1_score", 0)

    print("\nPRODUCTION DEPLOYMENT REVIEW")
    print("============================")
    print("Model:", MODEL_NAME)
    print("Version:", staging.version)
    print("F1 Score:", round(f1_score, 4))

    # Quality check
    if f1_score < MINIMUM_F1_SCORE:
        raise ValueError("Model failed quality validation")

    # Check validation tag
    tags = staging.tags

    if tags.get("validation_status") != "passed":
        raise ValueError("Model has not passed staging validation")

    # Check existing production
    try:
        current = client.get_model_version_by_alias(
            MODEL_NAME,
            "production"
        )

        print("Current production version:", current.version)

    except Exception:
        print("No existing production model.")

    # Manual approval
    approval = input(
        "\nType PROMOTE to approve production deployment: "
    )

    if approval != "PROMOTE":
        print("Deployment cancelled.")
        return

    # Promote model
    client.set_registered_model_alias(
        MODEL_NAME,
        "production",
        staging.version
    )

    client.set_model_version_tag(
        MODEL_NAME,
        staging.version,
        "deployment_status",
        "production"
    )

    print("\nSUCCESS!")
    print("Production model:", MODEL_NAME)
    print("Production version:", staging.version)


if __name__ == "__main__":
    promote_to_production()