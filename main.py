from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("myai.pkl")


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([{
        "area": data["area"],
        "bedrooms": data["bedrooms"],
        "bathrooms": data["bathrooms"],
        "stories": data["stories"],
        "mainroad": data["mainroad"],
        "parking": data["parking"],
        "prefarea": data["prefarea"],
        "furnishingstatus": data["furnishingstatus"]
    }])

    prediction = model.predict(input_data)

    return {
        "predicted_price": float(prediction[0])
    }