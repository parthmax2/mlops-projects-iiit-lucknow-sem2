import streamlit as st
from src.visualize import load_css
import os,sys



st.set_page_config(page_title="Stock Predictor", layout="wide")

load_css()

# Header
st.markdown("""
<div style="text-align: center; animation: fadeIn 2s;">
    <h1>📈 Stock Price Prediction App</h1>
    <p style="font-size: 1.1rem;">Select a stock to analyze and predict future prices</p>
</div>
""", unsafe_allow_html=True)

# Ticker Selection
with st.container():
    st.markdown('<div class="ticker-container">', unsafe_allow_html=True)

    ticker_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA']
    selected_ticker = st.selectbox(
        "Choose a stock ticker:",
        ticker_list,
        index=0,
        format_func=lambda x: {
            'AAPL': '🍎 Apple Inc. (AAPL)',
            'GOOGL': '🔍 Alphabet Inc. (GOOGL)',
            'MSFT': '🪟 Microsoft Corp. (MSFT)',
            'AMZN': '📦 Amazon.com Inc. (AMZN)',
            'META': '👤 Meta Platforms Inc. (META)',
            'TSLA': '🚗 Tesla Inc. (TSLA)'
        }.get(x, x)
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Right-aligned button
col1, col2, col3 = st.columns([5, 1, 1])
with col3:
    if st.button("Continue ➡️"):
        st.session_state['selected_ticker'] = selected_ticker
        st.switch_page("pages/1_Predict_Next_7_Days.py")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 Stock Prediction App. This is IIITL project.", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
            

