from flask import Flask, jsonify, request,send_file
import joblib as jb
import pandas as pd
import os
import matplotlib.pyplot as plt
import io

from ui import target_date, company

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Stock Price Prediction API running successfull"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    company = data.get("company")
    target_date = data.get("date")
    target_date = pd.to_datetime(target_date)

    model_path = f"model/{company}_model.pkl"

    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found for company"}), 404

    model_data = jb.load(model_path)
    model = model_data["model"]
    last_differenced_date = pd.to_datetime(model_data["last_date"]) # Last date of the differenced series
    last_original_close = model_data["last_original_close"]

    # Calculate the number of steps to forecast. This is from the last differenced date.
    days = (target_date - last_differenced_date).days

    if days <= 0:
        return jsonify({"error": "Date must be after last available date in the training data"}), 400

    # Forecast the differenced values
    differenced_forecast = model.forecast(steps=days)

    # Generate future dates for the forecast
    future_dates = pd.date_range(
        start=last_differenced_date + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )



    # Reconstruct the original price series from the differenced forecast
    # Start with the last known original close price and add the forecasted differences cumulatively
    reconstructed_prices = [last_original_close]
    for diff_val in differenced_forecast:
        reconstructed_prices.append(reconstructed_prices[-1] + diff_val)

    # The first element is the last_original_close, so slice from the second element for predictions
    predicted_prices = reconstructed_prices[1:]

    # Format predictions for output
    predictions = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "predicted_close_price": round(float(price), 2)
        }
        for date, price in zip(future_dates, predicted_prices)
    ]

    # Find the prediction for the requested target_date
    predicted_price_on_target_date = None
    for p in predictions:
        if p["date"] == target_date.strftime("%Y-%m-%d"):
            predicted_price_on_target_date = p["predicted_close_price"]
            break

    return jsonify({
        "company": company,
        "requested_date": target_date.strftime("%Y-%m-%d"),
        "predicted_price_on_date": predicted_price_on_target_date,

    })

@app.route("/chart",methods=['GET'])
def chart():
    company=request.args.get('company')
    tdate=request.args.get('date')
    tdate=pd.to_datetime(tdate)

    model_path = f"model/{company}_model.pkl"
    model_data = jb.load(model_path)
    model = model_data["model"]
    last_date=model_data['last_date']
    last_price=model_data['last_original_close']

    steps = (tdate-last_date).days
    differenced_forecast = model.forecast(steps=steps)

    price = [last_price]
    for d in differenced_forecast:
        price.append(price[-1] + d)

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=steps,
        freq="D"
    )

    history_dict = model_data["history"]
    history_df = history_dict[company].copy()
    history_df.index = pd.to_datetime(history_df.index)
    history_df = history_df[history_df.index <= tdate]

    # JSON safe
    history_json = [
        {"date": idx.strftime("%Y-%m-%d"), "close": round(float(val), 2)}
        for idx, val in history_df["close"].items()
    ]
    # Convert forecast to JSON
    forecast_json = [
        {"date": date.strftime("%Y-%m-%d"), "forecast": round(float(price[i + 1]), 2)}
        for i, date in enumerate(future_dates)
    ]

    # return JSON
    return jsonify({"history":history_json,"forecast":forecast_json})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)