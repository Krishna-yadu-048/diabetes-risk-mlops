"""
schemas.py

Pydantic models for the /predict endpoint.
These drive both request validation and the auto-generated Swagger docs.

Feature descriptions match the original Pima Indians Diabetes dataset.
"""

from pydantic import BaseModel, Field


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(..., gt=0, le=300, description="Plasma glucose concentration (mg/dL)")
    BloodPressure: float = Field(..., gt=0, le=200, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., gt=0, le=100, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., gt=0, le=900, description="2-hour serum insulin (mu U/ml)")
    BMI: float = Field(..., gt=0, le=70, description="Body mass index (weight in kg / height in m²)")
    DiabetesPedigreeFunction: float = Field(..., gt=0, le=3, description="Diabetes pedigree function score")
    Age: int = Field(..., gt=0, le=120, description="Age in years")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Pregnancies": 2,
                    "Glucose": 138,
                    "BloodPressure": 62,
                    "SkinThickness": 35,
                    "Insulin": 0,
                    "BMI": 33.6,
                    "DiabetesPedigreeFunction": 0.127,
                    "Age": 47,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No Diabetes, 1 = Diabetes")
    prediction_label: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., description="Model confidence for the predicted class (0–1)")
    model_name: str = Field(..., description="Name of the model that made the prediction")
