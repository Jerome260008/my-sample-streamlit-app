import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('rf_flood_model.pkl')

# Function for predictions
def predict_flooding(rainfall, flow_rate, river_level):
    input_data = pd.DataFrame({
        'Rainfall (mm)': [rainfall],
        'Flow Rate (cms)': [flow_rate],
        'River Water Level (m)': [river_level]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Page title with center alignment
st.markdown("<h1 style='text-align: center; color: blue;'>Flood Prediction App</h1>", unsafe_allow_html=True)

# Create input fields in the center
col1, col2, col3 = st.columns(3)
with col2:
    rainfall = st.number_input('🌧️ Rainfall (mm)', min_value=0.0)
    flow_rate = st.number_input('🌊 Flow Rate (cms)', min_value=0.0)
    river_level = st.number_input('🏞️ River Water Level (m)', min_value=0.0)

# Center the button
center_button = st.columns([1, 1, 1])
with center_button[1]:  # This ensures the button is in the center
    if st.button('🔍 Predict Flood Depth'):
        flood_depth = predict_flooding(rainfall, flow_rate, river_level)
        st.markdown(f"<h2 style='text-align: center; color: red;'>Predicted Flood Depth: {flood_depth:.2f} meters</h2>", unsafe_allow_html=True)
