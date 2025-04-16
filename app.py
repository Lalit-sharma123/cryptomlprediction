import streamlit as st
import joblib
import numpy as np

# Load the model 
model=joblib.load('liquidity_model.pkl')
scaler=joblib.load('scaler.pkl')

st.title("Cryptocurrency Liquidity Prediction")

price=st.number_input('Price')
h1=st.number_input("1 hour change(%)")
h24=st.number_input("24 hour change(%)")
h7=st.number_input("7 days change(%)")
volume=st.number_input("24h volume")
market_cap=st.number_input("market cap")

# Feature engineering
volatility=(price*h1)/100 
rolling_avg_price=price
liquidity_ratio_input=volume/(market_cap +1)

# prepare the input data for prediction 
# Prepare the input data for prediction
input_data = np.array([price, h1, h24, h7, rolling_avg_price, volatility, volume]).reshape(1, -1)

#input_data=np.array([price,h1,h24,h7,volume,market_cap,volatile,rolling_avg_price,liquidity_ratio_input])
scaled_input_data=scaler.transform(input_data)


# Predict liquidity
if st.button("Predict Liquidity"):
    prediction = model.predict(scaled_input_data)
    st.success(f'Predicted Liquidity Ratio: {prediction[0]:.4f}')
