"""
main.py

FastAPI application for the Diabetes Risk Predictor.

Endpoints:
  GET  /                   → Prediction dashboard (Jinja2 HTML)
  POST /dashboard-predict  → Handles dashboard form submission
  POST /predict            → REST API (JSON in, JSON out)
  GET  /health             → Model load status
  GET  /docs               → Custom Swagger UI
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.schemas import DiabetesInput, PredictionResponse

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-risk-model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# ──────────────────────────────────────────────────────────────────────────────

# Module-level model holder — loaded once at startup
_model = None
_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Production model from MLflow when the server starts."""
    global _model, _model_loaded
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@Production"
    try:
        _model = mlflow.sklearn.load_model(model_uri)
        _model_loaded = True
        print(f"Model loaded from: {model_uri}")
    except Exception as e:
        _model_loaded = False
        print(f"WARNING: Could not load model — {e}")
        print("The /predict endpoint will return 503 until a model is promoted to Production.")
    yield


app = FastAPI(
    title="Diabetes Risk Predictor",
    description="Predicts diabetes risk from clinical measurements using a model trained on the Pima Indians Diabetes dataset.",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

# Static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def run_prediction(data: dict) -> dict:
    """Shared prediction logic used by both REST and dashboard endpoints."""
    if not _model_loaded or _model is None:
        raise RuntimeError("Model is not loaded. Promote a model to Production in MLflow first.")

    df = pd.DataFrame([data], columns=FEATURE_COLS)
    prediction = int(_model.predict(df)[0])
    confidence = float(_model.predict_proba(df)[0][prediction])

    return {
        "prediction": prediction,
        "prediction_label": "Diabetes Risk Detected" if prediction == 1 else "No Diabetes Risk",
        "confidence": round(confidence, 4),
        "model_name": MODEL_NAME,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if _model_loaded else "model_not_loaded",
        "model_name": MODEL_NAME,
        "model_alias": "Production",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: DiabetesInput):
    result = run_prediction(payload.model_dump())
    return PredictionResponse(**result)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "form_data": None, "error": None},
    )


@app.post("/dashboard-predict", response_class=HTMLResponse)
def dashboard_predict(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...),
):
    form_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
    }

    try:
        result = run_prediction(form_data)
        error = None
    except Exception as e:
        result = None
        error = str(e)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "form_data": form_data, "error": error},
    )
