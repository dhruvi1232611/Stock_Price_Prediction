import matplotlib.pyplot as plt
import pandas as pd
import joblib as jb
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import root_mean_squared_error
from pmdarima import auto_arima
import yfinance as yf
import numpy as np

company = "AAPL"   # Apple
ticker = yf.Ticker(company)

# Last 2 years daily data
df = ticker.history(period="2y")

df.reset_index(inplace=True)
df['Company'] = company

df.to_csv("data/stocks_last_2_years.csv", index=False)


df=pd.read_csv("data/stocks_last_2_years.csv")
print(df.head())
print(f"\nMissing Value Count:\n{df.isnull().sum()}")


print(f"Statistics:{df.describe()}")
df['Date']=pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)
df = df.sort_index()

ts=df['Close'].dropna()
ts_log = np.log(ts)


def adf_test(timeseries):
    print("Results of dickey fuller test")
    result = adfuller(timeseries)
    print(f"Statistics:{result[0]}")
    print(f"p-value:{result[1]}")

adf_test(ts_log)

train_size=int(len(ts_log)*0.8)
train,test=ts[:train_size],ts[train_size:]


model=auto_arima(
    train,
    seasonal=True,
    m=1,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    trend='t',
    stepwise=False,
    d=None,
)


forecast=model.predict(n_periods=len(test))
rmse=root_mean_squared_error(test,forecast)
print(f"root_mean_squared_error Score:{rmse}")

plt.plot(ts)
plt.show()

plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(test.index,forecast,label='Forecast')
plt.legend()
plt.title("StocK Market Forecast")
plt.show()

jb.dump(model,"data/model.pkl")
