import matplotlib.pyplot as plt
import pandas as pd
import joblib as jb
from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import root_mean_squared_error,r2_score
from pmdarima import auto_arima
import yfinance as yf


companies = ["AAPL"]

all_data = []

for c in companies:
    df = yf.Ticker(c).history(period="2y")
    df.reset_index(inplace=True)
    df["Company"] = c
    all_data.append(df)

final_df = pd.concat(all_data)
final_df.to_csv("data/stocks_last_2_years.csv", index=False)


df=pd.read_csv("data/stocks_last_2_years.csv")
print(df.head())
print(f"\nMissing Value Count:\n{df.isnull().sum()}")


company = "AAPL"
df = df[df["Company"] == company]

print(f"Statistics:{df.describe()}")
df['Date']=pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)
df = df.sort_index()

ts=df['Close'].dropna()


def adf_test(timeseries):
    print("Results of dickey fuller test")
    result = adfuller(timeseries)
    print(f"Statistics:{result[0]}")
    print(f"p-value:{result[1]}")

adf_test(ts)

train_size=int(len(ts)*0.8)
train,test=ts[:train_size],ts[train_size:]

#model=ARIMA(train,order=(5,1,0))
#model_fit=model.fit()

model=auto_arima(
    train,seasonal=False,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    start_p=1,
    start_Q=1,
    max_p=5,
    max_q=5,
    trend='t',
    stepwise=True
)


forecast=model.predict(n_periods=len(test))
rmse=root_mean_squared_error(test,forecast)
print(f"root_mean_squared_error Score:{rmse}")



plt.plot(train,label='Train')
plt.plot(test,label='Test')
plt.plot(test.index,forecast,label='Forecast')
plt.legend()
plt.title("StocK Market Forecast")
plt.show()

jb.dump(model,"data/model.pkl")
