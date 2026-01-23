import streamlit as st
import requests
import pandas as pd
import requests as res
import joblib as jb

API_URL = "http://172.20.7.10:5001/predict"

st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide",
    page_icon="📈"
)

st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #fff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0b1020;
    color: #fff;
}

/* Title */
h1 {
    font-size: 2.5rem;
    font-weight: 700;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #06b6d4);
    color: white;
    border: none;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight: 700;
}

/* Line Chart */
.css-1q8dd3e canvas {
    background: rgba(255,255,255,0.05) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Prediction Dashboard")


# Sidebar
st.sidebar.title("📌 Options")
company = st.sidebar.selectbox(
    "Select Company",
    ["INDIA VIX_minute", "NIFTY ALPHA 50_minute", "NIFTY AUTO_minute", "NIFTY BANK_minute", "NIFTY COMMODITIES_minute",
     "NIFTY CONSR DURBL_minute", "NIFTY CONSUMPTION_minute", "NIFTY CPSE_minute", "NIFTY ENERGY_minute", "NIFTY FIN SERVICE_minute",
     "NIFTY FMCG_minute", "NIFTY GS COMPSITE_minute", "NIFTY HEALTHCARE_minute", "NIFTY IND DIGITAL_minute", "NIFTY INDIA MFG_minute",
     "NIFTY INFRA_minute", "NIFTY IT_minute", "NIFTY LARGEMID250_minute"]
)
target_date = st.sidebar.date_input("Select Date")


if st.sidebar.button("Predict"):

    payload = {
        "company": company,
        "date": str(target_date)
    }

    with st.spinner("Fetching prediction..."):
        res = requests.post(API_URL, json=payload)

    if res.status_code != 200:
        st.error("API Error")
    else:
        data = res.json()

        st.metric(
            label=f"{company} Predicted Price",
            value=f"₹ {data['predicted_price_on_date']}"
        )

    st.metric("Predicted Price","Fetched")

    chart_url=f"http://127.0.0.1:5001/chart?company={company}&date={target_date}"
    #st.image(chart_url,caption=f"{company} price trend")



    res = requests.get(chart_url)
    if res.status_code == 200:
        data = res.json()

        hist_df = pd.DataFrame(data["history"])
        fore_df = pd.DataFrame(data["forecast"])

        # Convert date column to datetime
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        fore_df["date"] = pd.to_datetime(fore_df["date"])

        # Merge on date
        df = pd.merge(hist_df, fore_df, on="date", how="outer")

        df = df.set_index("date")

        st.subheader(f"{company}'s Trend Stock Price")

        st.line_chart(df,x_label="Date",y_label="close_price",color=["#FF0000", "#0000FF"])



