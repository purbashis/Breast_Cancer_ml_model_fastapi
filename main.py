
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

#pydantic is used for data validation and settings management using python type annotations
from pydantic import BaseModel

#joblib is used for saving and loading machine learning models
import joblib
import numpy as np

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# load model & scaler

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title = "Breast Cancer Prediction API")



#here Cancer input is defined with 30 float features
class CancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float



 
#here Cancer output is defined with prediction and probability   
class CancerOutput(BaseModel):
    prediction: str
    probability: float
    
@app.post("/predict", response_model=CancerOutput)
async def predict(data: CancerInput):
    values = [
    data.mean_radius,
    data.mean_texture,
    data.mean_perimeter,
    data.mean_area,
    data.mean_smoothness,
    data.mean_compactness,
    data.mean_concavity,
    data.mean_concave_points,
    data.mean_symmetry,
    data.mean_fractal_dimension,

]


    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0].max()

    return {
        "prediction": "Benign" if pred == 1 else "Malignant",
        "probability": float(prob)
    }


app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
