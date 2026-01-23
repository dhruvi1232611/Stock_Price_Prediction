from flask import Flask, request, jsonify # Import jsonify here
from flask_restful import Api, Resource
from flasgger import Swagger, swag_from
#from pyngrok import ngrok
import joblib as jb
import pandas as pd
import os

#ngrok.set_auth_token("38bHkS4PR0l7lqLtJHlRhwGMPhv_3v7KazZKJmpvfunncWYq8")

app=Flask(__name__)
api=Api(app)

app.config['SWAGGER']={
    'title':'My API',
    'uiversion':3
}
swagger=Swagger(app)

class welcome(Resource):
    @swag_from({
        'responses':{
            200:{
                'description':'A status 200 means successful and return a message',
                'content':{
                    'application/json':{
                        'examples':{
                            'example1':{
                                'summary':'sucessfull',
                                'value':{
                                    'message':'welcome'
                                }
                            }
                        }
                    }
                }
            }
        }
    })
    def get(self):
        """
        This is an example endpoint which returns a simple message.
        """
        return {'message':'welcome'}

# Removed the 'post' method from 'welcome' class as it's handled by the @app.route('/price') below

@app.route('/price',methods=['POST'])
@swag_from({
    'summary': 'Predicts stock price for a given company and date',
    'parameters': [
        {
            'in': 'body',
            'name': 'body',
            'schema': {
                'type': 'object',
                'required': [
                    'company',
                    'date'
                ],
                'properties': {
                    'company': {
                        'type': 'string',
                        'example': 'NIFTY AUTO_minute'
                    },
                    'date': {
                        'type': 'string',
                        'format': 'YYYY-MM-DD',
                        'example': '2025-08-01'
                    }
                }
            },
            'description': 'Input data for prediction'
        }
    ],
    'responses': {
        200: {
            'description': 'Successful prediction',
            'schema': {
                'type': 'object',
                'properties': {
                    'company': {'type': 'string'},
                    'requested_date': {'type': 'string', 'format': 'YYYY-MM-DD'},
                    'predicted_price_on_date': {'type': 'number'}
                }
            },
            'examples': {
                'prediction_example': {
                    'company': 'NIFTY AUTO_minute',
                    'requested_date': '2025-08-01',
                    'predicted_price_on_date': 12345.67
                }
            }
        },
        400: {
            'description': 'Bad request - Date must be after last available date',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        },
        404: {
            'description': 'Model not found for company',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    }
})
def price():
    data=request.get_json()
    company = data.get("company")
    target_date = data.get("date") # Renamed 'date' to 'target_date' to avoid confusion
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

    predictions=[
        {
            'date':date.strftime('%Y-%m-%d'),
            'predicted_close_price':round(float(price),2)
        }
        for date,price in zip(future_dates,predicted_prices)
    ]
    predicted_price_on_target_date=None
    for p in predictions:
        if p['date']==target_date.strftime('%Y-%m-%d'): # Corrected variable name from 'date' to 'target_date'
            predicted_price_on_target_date=p['predicted_close_price']
            break
    return jsonify( # Added jsonify and corrected the return structure
    {
        'company':company,
        'requested_date':target_date.strftime('%Y-%m-%d'), # Corrected variable name from 'date' to 'target_date'
        'predicted_price_on_date':predicted_price_on_target_date
    })

api.add_resource(welcome,'/') # Only add 'welcome' to the root path
if __name__ == '__main__':
    #public_url = ngrok.connect(5000)
    #print("Public URL:", public_url)
    app.run(debug=True,use_reloader=False)