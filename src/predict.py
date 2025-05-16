import numpy as np


#----------------------------------------
# Get Recommendation
#-----------------------------------------

def get_recommendation(predicted_prices, last_price):
    """Generate buy/sell/hold recommendation based on price prediction"""
    first_day_price = predicted_prices[0]
    seventh_day_price = predicted_prices[-1]
    
    # Calculate expected returns
    long_term_return = (seventh_day_price - last_price) / last_price * 100
    
    # Recommendation logic
    if long_term_return > 5:
        return "BUY", "Strong upward trend predicted over the next week", "#dcfce7", "#166534"
    elif long_term_return > 2:
        return "BUY", "Moderate upward trend predicted", "#dcfce7", "#166534"
    elif long_term_return < -5:
        return "SELL", "Strong downward trend predicted over the next week", "#fee2e2", "#991b1b"
    elif long_term_return < -2:
        return "SELL", "Moderate downward trend predicted", "#fee2e2", "#991b1b"
    else:
        return "HOLD", "No significant price movement predicted", "#fef9c3", "#854d0e"




#----------------------------------------
# Predict Next Day Single Feature
#-----------------------------------------

def predict_next_days_single_feature(model, x_input, scaler, time_step=60, days=7):
    """
    Predict stock prices for the next 7 days using a model trained on only the Close price.
    """
    try:
        # Array to store predictions
        predicted_prices = []
        
        # Make a copy of the input data to avoid modifying the original
        current_batch = x_input.copy()
        
        # Loop for each day we want to predict
        for i in range(days):
            # Make prediction for the current batch (shape will be [1, 1])
            current_pred = model.predict(current_batch, verbose=0)
            
            # Store the prediction
            predicted_prices.append(current_pred[0, 0])
            
            # Update the input sequence for the next prediction
            # Remove the oldest data point and add the new prediction
            new_point = np.array([[[current_pred[0, 0]]]])  # Shape: [1, 1, 1]
            current_batch = np.append(current_batch[:, 1:, :], new_point, axis=1)
        
        # Convert the scaled predictions back to original price scale
        predicted_prices_array = np.array(predicted_prices).reshape(-1, 1)
        unscaled_predictions = scaler.inverse_transform(predicted_prices_array)
        
        # Return as a flat array
        return unscaled_predictions.flatten()
    except Exception as e:
        print(f"Error predicting next days: {str(e)}")
        return None





