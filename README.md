Stock Price Prediction 

This project is an end-to-end machine learning solution designed to forecast Nifty 50 stock prices. 
It leverages the ARIMA (Autoregressive Integrated Moving Average) statistical model to analyze historical time-series data and predict future trends.

Collect the dataset from kaggle 
Tech Stack:
Language: Python
Modeling: Statsmodels (ARIMA), Pandas, NumPy
Frontend: Streamlit
Backend: Flask
API Docs: Swagger / Flasgger

Project Workflow
Data Preprocessing: Historical Nifty 50 data is cleaned and checked for stationarity using the Augmented Dickey-Fuller (ADF) test.

Model Training: The ARIMA(p, d, q) parameters are optimized to fit the historical closing prices.

Serialization: The best-performing model is saved as model.pkl.



How to run the model

step 1 : Run the streamlit ui 
streamlit run ui.py

step 2 : Run the flsk api 
python app.py

for api documentation : swagger url
http://172.20.7.19:5000/apidocs
