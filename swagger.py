from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Initialize FastAPI
app = FastAPI(
    title="Future Stock Price Prediction API",
    description="Predict future stock prices using ARIMA",
    version="1.0"
)

# Load trained ARIMA model
model = joblib.load("data/model.pkl")


# Request schema
class PredictionRequest(BaseModel):
    company:str
    days: int


# Response schema
class PredictionResponse(BaseModel):
    company:str
    forecast_days: int
    predicted_prices: list


@app.post("/predict", response_model=PredictionResponse)
def predict_future_prices(request: PredictionRequest):

    if request.days <= 0:
        raise HTTPException(status_code=400, detail="Days must be greater than 0")

    try:
        # auto_arima uses predict(n_periods)
        forecast = model.predict(n_periods=request.days)

        predictions = [
            {
                "day": i + 1,
                "predicted_close_price": round(float(price), 2)
            }
            for i, price in enumerate(forecast)
        ]

        return {
            "company":request.company,
            "forecast_days": request.days,
            "predicted_prices": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return "Future Stock Forecast API is running"
