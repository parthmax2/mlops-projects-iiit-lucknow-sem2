# Air Quality Index (AQI) Forecasting System

A Python-based system for analyzing and forecasting air quality using time series decomposition and SARIMAX modeling. It includes both command-line and web-based (Streamlit) interfaces to generate date-specific forecasts, identify dominant pollutants, and classify AQI based on EPA standards.

---

## Features

### Core Functionality
- AQI forecasting using historical pollutant data  
- AQI categorization using EPA standards (Good to Hazardous)  
- Identification of dominant pollutants  
- Date-specific forecasts via CLI and Web UI  

### Advanced Analytics
- Automatic handling of missing values (forward fill and spline interpolation)  
- STL decomposition (Seasonal-Trend using LOESS)  
- SARIMAX modeling with exogenous inputs and automated parameter tuning  
- Forecast evaluation using MAE, RMSE, and R² metrics  
- Visualization of trends, seasonality, and forecast results  

---

## Installation

### Requirements
- Python 3.7 or above  
- Required Libraries:  
  `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`, `scikit-learn`, `streamlit`, `joblib`

### Install Dependencies
```bash
pip install pandas numpy statsmodels scipy matplotlib scikit-learn streamlit joblib
