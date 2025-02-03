import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load your model
model = joblib.load('rf_flood_model.pkl')

# Create a function to make predictions
def predict_flooding(rainfall, flow_rate, river_level):
    input_data = pd.DataFrame({
        'Rainfall (mm)': [rainfall],
        'Flow Rate (cms)': [flow_rate],
        'River Water Level (m)': [river_level]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit interface
st.title('Flood Prediction App')

st.sidebar.header('User Input')
rainfall = st.sidebar.number_input('Rainfall (mm)', min_value=0.0)
flow_rate = st.sidebar.number_input('Flow Rate (cms)', min_value=0.0)
river_level = st.sidebar.number_input('River Water Level (m)', min_value=0.0)

if st.sidebar.button('Predict Flood Depth'):
    flood_depth = predict_flooding(rainfall, flow_rate, river_level)
    st.write(f'Predicted Flood Depth: {flood_depth} meters')