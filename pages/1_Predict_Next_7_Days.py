from src.download_data import *
from src.preprocess import *
from src.predict import *
from src.visualize import *
from src.file_handling import *

import streamlit as st
import pandas as pd
import numpy as np
import time

from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Prediction", layout="wide")

load_css()

# App header
st.markdown("<h1 style='text-align: center'>🔮 Stock Price Prediction</h1>", unsafe_allow_html=True)

ticker = st.session_state.get('selected_ticker', None)

if ticker:
    st.markdown(f"""
    <div class="card">
        <h2>Selected Stock: {ticker}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    prediction_placeholder = st.empty()
    
    if st.button("Run Prediction"):
        with st.spinner("Running prediction pipeline..."):
            # Show a progress bar
            progress_bar = st.progress(0)
            progress_bar.progress(10)
            time.sleep(0.5)
            
            if should_download(ticker):
                # Delete Existing Files 
                cleanup_old_ticker_files(ticker)
                # Download and preprocess data
                data_path = download_stock_data(ticker)
                progress_bar.progress(30)
                time.sleep(0.5)
            
                recent_path = preprocess_data(data_path, ticker)
                progress_bar.progress(50)
                time.sleep(0.5)
            else:
                print("⏳ Skipping download.")
                recent_path = f"data/{ticker}_clean.csv"
               
            recent_df = pd.read_csv(recent_path)
            progress_bar.progress(60)
            
            # Fix columns
            if 'Date' not in recent_df.columns:
                recent_df.reset_index(inplace=True)
                recent_df.rename(columns={'index': 'Date'}, inplace=True)
            recent_df['Date'] = pd.to_datetime(recent_df['Date'])
            
            historical_df = load_historical_data(ticker)
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            progress_bar.progress(70)
            
            all_data = historical_df
            
            # Prepare data for model
            x_input, scaler, _ = prepare_data_for_single_feature_model(all_data)
            progress_bar.progress(80)
            time.sleep(0.5)
            
            model_path = f"model/{ticker}_model.h5"
            
            # Check if model exists and make prediction
            if os.path.exists(model_path):
                model = load_model(model_path)
                predicted_prices = predict_next_days_single_feature(model, x_input, scaler)
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Remove progress bar
                progress_bar.empty()
                
                # Display results if prediction was successful
                if predicted_prices is not None:
                    last_price = all_data['Close'].iloc[-1]
                    
                    # Display the prediction results in the placeholder
                    with prediction_placeholder.container():
                        # Create columns for the layout
                        rec_col, chart_col = st.columns([1, 2])
                        
                        # Get recommendation
                        rec, reason, bg_color, text_color = get_recommendation(predicted_prices, last_price)
                        
                        # Show recommendation in the left column
                        with rec_col:
                            st.markdown(f"""
                            <div class="card" style="height: 100%;">
                                <h3 style="text-align: center;">Recommendation</h3>
                                <div style="background-color: {bg_color}; color: {text_color}; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px;">
                                    <h1 style="font-size: 2.2rem; margin-bottom: 10px;">{rec}</h1>
                                    <p style="color:{text_color}">{reason}</p>
                                </div>
                                <div style="margin-top: 20px;">
                                    <p><strong>Last Price:</strong> ${last_price:.2f}</p>
                                    <p><strong>Predicted (7 days):</strong> ${predicted_prices[-1]:.2f}</p>
                                    <p><strong>Potential Return:</strong> {((predicted_prices[-1] - last_price) / last_price * 100):.2f}%</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        

                        # Show chart in the right column
                        with chart_col:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            fig = create_prediction_chart(all_data, predicted_prices)
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show prediction table below
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📊 Detailed Price Predictions")
                        pred_df = create_prediction_table(predicted_prices)
                        st.dataframe(pred_df, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add analysis notes
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📝 Analysis Summary")
                        
                        # Calculate some statistics
                        min_price = min(predicted_prices)
                        max_price = max(predicted_prices)
                        volatility = np.std(predicted_prices) / np.mean(predicted_prices) * 100
                        
                        st.markdown(f"""
                        - The model predicts a {'positive' if predicted_prices[-1] > last_price else 'negative'} trend for {ticker} over the next 7 trading days.
                        - Predicted price range: ${min_price:.2f} to ${max_price:.2f}
                        - Expected volatility: {volatility:.2f}%
                        
                        **Disclaimer:** These predictions are based on historical patterns and should not be the sole basis for investment decisions.
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Model not found for {ticker}. Please try a different stock.")
else:
    st.warning("Please select a ticker from the Home page.")

# Add a navigation menu
st.sidebar.markdown("### Navigation")
if st.sidebar.button("🏠 Home"):
    st.switch_page("app.py")
if st.sidebar.button("📊 Historical Data"):
    st.switch_page("pages/2_View_Monthly_Yearly.py")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 Stock Prediction App. This is for educational purposes only.", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)