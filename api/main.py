from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# -------------------------
# 1. Load model + features
# -------------------------
MODEL_PATH = "models/best_model.pkl"
FEATURES_PATH = "models/features.pkl"

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)

# -------------------------
# 2. App init
# -------------------------
app = FastAPI(title="House Prices API", version="1.0")

# -------------------------
# 3. Input schema
# -------------------------
class HouseFeatures(BaseModel):
    features: dict

# -------------------------
# 4. Health check
# -------------------------
@app.get("/")
def health():
    return {"status": "API running"}

# -------------------------
# 5. Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(data: HouseFeatures):

    # convertir input a dataframe
    df = pd.DataFrame([data.features])

    # Alinear con features del entrenamiento
    df = df.reindex(columns=FEATURES, fill_value=0)

    prediction = model.predict(df)[0]

    return {
        "prediction": float(prediction)
    }