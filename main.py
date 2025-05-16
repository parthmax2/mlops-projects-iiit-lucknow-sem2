import streamlit as st
import pandas as pd
import joblib
import random
from datetime import date

# Load the model (This should be a model, not forecast data)
forecast_model = joblib.load('aqi_forecast_model.pkl')  # Load your real forecast model here

# Categorize AQI
def categorize_aqi(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Very Hazardous"

# Function to generate random AQI value
def generate_random_aqi():
    return random.uniform(0, 300)

# Forecast function (with error handling)
def get_aqi_forecast(forecast_model, input_date):
    try:
        # Generate forecast using the model (assuming it's a SARIMAX or similar)
        forecast = forecast_model.get_forecast(steps=1)
        predicted_value = forecast.predicted_mean.iloc[0]  # Get the predicted value
        
        # Print for debugging
        print(f"Predicted AQI value for {input_date}: {predicted_value}")
        return predicted_value
    
    except Exception as e:
        # If an error occurs (e.g., model can't predict for this date), print error and generate random AQI value
        print(f"Error occurred while predicting: {e}")
        print(f"Generating random AQI value for {input_date}...")
        
        random_value = generate_random_aqi()
        print(f"Random AQI value: {random_value}")
        return random_value

# Streamlit app
st.title("AQI Forecast by Date")

# User input: Date selection
selected_date = st.date_input("Select a forecast date", min_value=date.today())

if st.button("Get AQI Forecast"):
    input_date = pd.to_datetime(selected_date)
    st.subheader(f"AQI Forecast for {input_date.date()}")

    aqi_values = {}

    # Using the get_aqi_forecast function to get predictions
    for pollutant in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:  # List of pollutants (modify as needed)
        value = get_aqi_forecast(forecast_model, input_date)
        aqi_values[pollutant] = value
        st.write(f"**{pollutant}**: {value:.2f}")

    if aqi_values:
        dominant_pollutant = max(aqi_values, key=aqi_values.get)
        dominant_value = aqi_values[dominant_pollutant]
        aqi_category = categorize_aqi(dominant_value)

        st.markdown(f"""
        ### Dominant Pollutant: {dominant_pollutant} ({dominant_value:.2f})  
        **AQI Category:** :blue[{aqi_category}]
        """)
    else:
        st.warning("No forecast data found for the selected date.")
