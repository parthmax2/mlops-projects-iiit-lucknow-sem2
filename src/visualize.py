from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

#------------------------------------
# Load CSS
#------------------------------------
def load_css():
    with open('style.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)



#----------------------------------------
# Create Prediction Chart
#----------------------------------------
def create_prediction_chart(historical_data, predicted_prices):
    """Create an interactive chart with historical and predicted prices"""
    # Get the last date from historical data
    last_date = historical_data['Date'].iloc[-1]
    
    # Generate dates for the next 7 days (excluding weekends)
    future_dates = []
    date = last_date - timedelta(days=1)
    days_added = 0
    while days_added < 7:
        date = date + timedelta(days=1)
        if date.weekday() < 5:  # Monday to Friday
            future_dates.append(date)
            days_added += 1
        
    # Create a dataframe for the predicted prices
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted': predicted_prices
    })
    
    # Create a figure
    fig = go.Figure()
    
    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=historical_data['Date'].iloc[-30:],  # Show last 30 days
        y=historical_data['Close'].iloc[-30:],
        mode='lines',
        name='Historical Price',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Add predicted data trace
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Predicted'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#10b981', width=3, dash='dash'),
        marker=dict(size=8, symbol='circle')
    ))
    
    # Add hover data
    fig.update_traces(
        hoverinfo="text+name",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:.2f}<extra></extra>"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


#----------------------------------------
# Create Prediction Table
#----------------------------------------

def create_prediction_table(predicted_prices):
    """Create a table with prediction dates and prices"""
    last_date = datetime.now()-timedelta(days=1)
    future_dates = []
    date = last_date
    days_added = 0
    while days_added < 7:
        date = date + timedelta(days=1)
        if date.weekday() < 5:  # Monday to Friday
            future_dates.append(date.strftime('%Y-%m-%d'))
            days_added += 1
    
    # Create a dataframe for the predicted prices
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price ($)': [f"${price:.2f}" for price in predicted_prices],
        'Change (%)': [(price - predicted_prices[0]) / predicted_prices[0] * 100 if i > 0 else 0 for i, price in enumerate(predicted_prices)]
    })
    
    # Format the change column
    pred_df['Change (%)'] = pred_df['Change (%)'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%" if x < 0 else "0.00%")
    
    return pred_df


#----------------------------------------
# Create Price Chart
#----------------------------------------

def create_price_chart(df):
    """Create a simple line chart for stock price with volume as bars"""
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name="Price",
            line=dict(color='#3b82f6', width=2)
        )
    )
    
    # Add moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MA5'],
            mode='lines',
            name="5-day MA",
            line=dict(color='#f59e0b', width=2, dash='dot')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MA20'],
            mode='lines',
            name="20-day MA",
            line=dict(color='#ef4444', width=2, dash='dot')
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add range selector
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor="#e2e8f0",
                activecolor="#3b82f6"
            ),
            type="date"
        )
    )
    
    return fig



#-------------------------------------
# Calculate Stats
#-------------------------------------

def calculate_stats(df):
    """Calculate key statistics for the selected period"""
    first_price = df['Open'].iloc[0]
    last_price = df['Close'].iloc[-1]
    highest_price = df['High'].max()
    lowest_price = df['Low'].min()
    avg_price = df['Close'].mean()
    price_change = last_price - first_price
    price_change_pct = (price_change / first_price) * 100
    daily_returns = df['Close'].pct_change().dropna()
    volatility = daily_returns.std() * 100  # Daily volatility
    
    return {
        'first_price': first_price,
        'last_price': last_price,
        'highest_price': highest_price,
        'lowest_price': lowest_price,
        'avg_price': avg_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'volatility': volatility
    }





