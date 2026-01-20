from flask import Flask,request, jsonify
import joblib as jb


app=Flask(__name__)
model=jb.load("data/model.pkl")

@app.route("/predict",methods=["POST"])
def predict():
    data=request.get_json()
    company=data.get("Company")
    day=data.get("days")

    forecast=model.predict(n_periods=int(day))

    price_prediction = [
        {
            "day": i + 1,
            "predicted_close_price": round(float(price), 2)
        }
        for i, price in enumerate(forecast)
    ]

    return jsonify({
        "company": company,
        "forecast_days": day,
        "Price": price_prediction
    })

@app.route("/")
def home():
    return jsonify({"message": "ARIMA Stock Prediction API running"})

if __name__ == "__main__":
    app.run(debug=True)