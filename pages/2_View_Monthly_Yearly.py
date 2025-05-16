import streamlit as st
from src.download_data import load_historical_data
from src.visualize import *



st.set_page_config(page_title="Historical Stock Data", layout="wide")

load_css()

# App header
st.markdown("<h1 style='text-align: center'>📅 Historical Stock Data</h1>", unsafe_allow_html=True)

ticker = st.session_state.get('selected_ticker', None)

# Helper functions
def get_month_name(month_num):
    return datetime(2024, month_num, 1).strftime('%B')


if ticker:
    st.markdown(f"""
    <div class="card">
        <h2>Historical Data for {ticker}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load historical data
    df = load_historical_data(ticker)
    if df is not None and not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))
        with col2:
            month = st.selectbox("Select Month", sorted(df[df['Year'] == year]['Month'].unique()), 
                               format_func=get_month_name)
        
        # Filter the data
        filtered_df = df[(df['Year'] == year) & (df['Month'] == month)]
        
        if not filtered_df.empty:
            # First show the chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            fig = create_price_chart(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calculate statistics
            stats = calculate_stats(filtered_df)
            
            # Display statistics cards
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Key Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Period Change</h4>
                    <p class="{'up-value' if stats['price_change'] >= 0 else 'down-value'}">
                        {stats['price_change_pct']:.2f}% 
                        ({'+' if stats['price_change'] >= 0 else ''}{stats['price_change']:.2f} USD)
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Price Range</h4>
                    <p>
                        ${stats['lowest_price']:.2f} - ${stats['highest_price']:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Average Price</h4>
                    <p>
                        ${stats['avg_price']:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="stat-card">
                    <h4>Daily Volatility</h4>
                    <p>
                        {stats['volatility']:.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Show the table of filtered data
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"📅 Stock Data for {get_month_name(month)} {year}")
            display_df = filtered_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No data available for the selected month and year.")
    else:
        st.error("Historical data could not be loaded or is empty.")
else:
    st.warning("Please select a ticker from the main page.")

# Add a navigation menu
st.sidebar.markdown("### Navigation")
if st.sidebar.button("🏠 Home"):
    st.switch_page("streamlit_app.py")
if st.sidebar.button("🔮 Predictions"):
    st.switch_page("pages/1_Predict_Next_7_Days.py")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("© 2025 Stock Prediction App. This is for educational purposes only.", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
            



            
