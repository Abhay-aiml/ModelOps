from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from monitoring.performance_monitor import (
    record_ground_truth,
    calculate_performance
)
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

from monitoring.prediction_analytics import get_prediction_analytics

from monitoring.monitoring_service import (
    log_prediction,
    run_drift_analysis,
    get_latest_report,
    get_monitoring_status
)


app = FastAPI(
    title="ModelOps API",
    description="ML model serving and monitoring platform",
    version="2.0.0"
)


# ==========================================
# MLFLOW CONFIGURATION
# ==========================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "ModelOpsClassifier"
MODEL_ALIAS = "production"

model = None
model_version = None


# ==========================================
# FEATURE NAMES
# ==========================================

FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# ==========================================
# REQUEST SCHEMAS
# ==========================================

class PredictionInput(BaseModel):
    features: dict[str, float]
class GroundTruthInput(BaseModel):
    prediction_id: str
    actual_label: int    


# ==========================================
# LOAD PRODUCTION MODEL
# ==========================================

@app.on_event("startup")
def load_production_model():

    global model, model_version

    client = mlflow.MlflowClient()

    # Get the current production model
    version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    model_version = version.version

    # Get experiment ID
    run = client.get_run(version.run_id)
    experiment_id = run.info.experiment_id

    # Extract logged model ID
    model_id = version.source.rsplit("/", 1)[-1]

    # Build local artifact path
    model_path = (
        Path(__file__).resolve().parents[1]
        / "mlruns"
        / experiment_id
        / "models"
        / model_id
        / "artifacts"
    )

    # Verify model files exist
    if not model_path.exists():
        raise FileNotFoundError(
            f"Production model artifacts not found: {model_path}"
        )

    # Load model
    model = mlflow.sklearn.load_model(str(model_path))

    print(
        f"Loaded {MODEL_NAME} version {model_version}"
    )


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "ModelOps API is running",
        "status": "healthy"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model": MODEL_NAME,
        "model_version": model_version
    }


# ==========================================
# MODEL PREDICTION
# ==========================================

@app.post("/predict")
def predict(data: PredictionInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    missing = set(FEATURE_NAMES) - set(data.features)

    extra = set(data.features) - set(FEATURE_NAMES)

    if missing or extra:

        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing),
                "unexpected_features": sorted(extra)
            }
        )

    try:

        input_data = pd.DataFrame(
            [[data.features[name] for name in FEATURE_NAMES]],
            columns=FEATURE_NAMES
        )

        prediction = model.predict(input_data)

        probabilities = model.predict_proba(input_data)

        # Save prediction and generate unique ID
        prediction_log = log_prediction(
            features=data.features,
            prediction=prediction[0],
            probabilities={
                "malignant": probabilities[0][0],
                "benign": probabilities[0][1]
            },
            model_version=model_version
        )

        return {
            "prediction_id": prediction_log["prediction_id"],

            "prediction": int(prediction[0]),

            "prediction_label": (
                "malignant"
                if int(prediction[0]) == 0
                else "benign"
            ),

            "probabilities": {
                "malignant": float(probabilities[0][0]),
                "benign": float(probabilities[0][1])
            },

            "model_version": model_version
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ==========================================
# MONITORING STATUS
# ==========================================

@app.get("/monitoring/status")
def monitoring_status():

    try:

        return get_monitoring_status()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# DATA DRIFT REPORT
# ==========================================

@app.get("/monitoring/drift")
def monitoring_drift():

    try:

        return get_latest_report()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# RUN DRIFT ANALYSIS
# ==========================================

@app.post("/monitoring/analyze")
def analyze_drift():

    try:

        report = run_drift_analysis()

        return {
            "message": "Drift analysis completed successfully",
            "report": report
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# PREDICTION ANALYTICS
# ==========================================

@app.get("/monitoring/analytics")
def prediction_analytics():

    try:

        return get_prediction_analytics()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # ==========================================
# GROUND TRUTH COLLECTION
# ==========================================

@app.post("/monitoring/ground-truth")
def submit_ground_truth(data: GroundTruthInput):

    try:
        return record_ground_truth(
            prediction_id=data.prediction_id,
            actual_label=data.actual_label
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# MODEL PERFORMANCE MONITORING
# ==========================================

@app.get("/monitoring/performance")
def model_performance():

    try:
        return calculate_performance()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )