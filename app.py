from fastapi import FastAPI
import joblib
import pandas as pd
from schemas import HouseInput
from sklearn.preprocessing import *
from sklearn.compose import ColumnTransformer

app = FastAPI(
    title="House Price Prediction API",
    version="1.0"
)

# Load model once
model = joblib.load("house_price_model.pkl")
print(model.n_features_in_)

@app.get("/")
def home():
    return {"message": "House Price Prediction API running"}

@app.post("/predict")
def predict_price(data: HouseInput):
    df = pd.DataFrame([{
        "area": data.area,
        "bedrooms": data.bedrooms,
        "bathrooms": data.bathrooms,
        "stories": data.stories,
        "mainroad": data.mainroad,
        "guestroom": data.guestroom,
        "basement": data.basement,
        "hotwaterheating": data.hotwaterheating,
        "airconditioning": data.airconditioning,
        "parking": data.parking,
        "prefarea": data.prefarea,
        "furnishingstatus": data.furnishingstatus,
        "location_type": data.location_type,
        "base_ppsf": data.base_ppsf
    }])

    
    prediction = model.predict(df)[0]

    return {"predicted_price": round(float(prediction), 2)}