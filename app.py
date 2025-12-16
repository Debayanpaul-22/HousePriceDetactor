from fastapi import FastAPI
import joblib
import pandas as pd
from schemas import HouseInput
from sklearn.preprocessing import *
from sklearn.compose import ColumnTransformer
from fastapi import HTTPException
import traceback


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
    try:
        df = pd.DataFrame([{
            "area": data.area,
            "bedrooms": data.bedrooms,
            "bathrooms": data.bathrooms,
            "stories": data.stories,
            "mainroad": data.mainroad.lower(),
            "guestroom": data.guestroom.lower(),
            "basement": data.basement.lower(),
            "hotwaterheating": data.hotwaterheating.lower(),
            "airconditioning": data.airconditioning.lower(),
            "parking": data.parking,
            "prefarea": data.prefarea.lower(),
            "furnishingstatus": data.furnishingstatus.lower(),
            "location_type": data.location_type.lower(),
            "base_ppsf": data.base_ppsf
        }])
    
    

    
        prediction = model.predict(df)[0]

        return {"predicted_price": round(float(prediction), 2)}
    
    except Exception as e:
        traceback.print_exc()  # 🔥 shows real error in Render logs
        raise HTTPException(status_code=500, detail=str(e))