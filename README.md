# 📈 Stock Price Prediction using LSTM

This project provides an end-to-end pipeline for **predicting stock prices** using LSTM (Long Short-Term Memory) deep learning models. It fetches historical stock data from **2004 to the present**, trains separate models for each stock ticker, and provides a user-friendly **Streamlit app** to visualize, predict, and interact with the results.

---

## 🚀 Features

- 📥 Download and merge historical + recent stock data
- 🧼 Clean & preprocess data (scaling, formatting)
- 🧠 Train LSTM models per ticker (AAPL, GOOGL, AMZN, TSLA, MSFT, META)
- 🔮 Predict the next **7 days** of stock prices using trained models
- 📊 Visualize historical, monthly, and yearly stock trends
- 📤 Export data to CSV
- 📱 Interactive Streamlit UI with **Buy / Sell / Hold** recommendations

---

## 🧠 LSTM Model Architecture

```python
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

```
## 📈 Metrics

| Ticker | MSE    | RMSE  | MAE  | R² Score |
| ------ | ------ | ----- | ---- | -------- |
| AAPL   | 26.62  | 5.16  | 4.04 | 0.9742   |
| TSLA   | 15.32  | 3.91  | 3.04 | 0.9851   |
| AMZN   | 21.44  | 4.63  | 3.51 | 0.9786   |
| GOOGL  | 16.96  | 4.12  | 3.18 | 0.9745   |
| META   | 116.26 | 10.78 | 7.55 | 0.9954   |
| MSFT   | 69.55  | 8.34  | 6.56 | 0.9858   |


## 🗂️ Project Structure

```
Stock-Price-Prediction-using-LSTM/
├── .dvc/                     # DVC config and tracking
├── combine_data/            # Merged past & recent stock data
├── data/                    # Raw downloaded stock data
├── historical/              # Original historical stock data (2004–2024)
├── model/                   # Trained models (.h5)
│   ├── AAPL_model.h5
│   ├── GOOGL_model.h5
│   └── ...
├── notebooks/               # Jupyter notebooks for exploration
├── pages/                   # Streamlit pages
│   ├── 1_Predict_Next_7_Days.py
│   └── 2_View_Monthly_Yearly.py
├── rough/                   # Backup or experimental models/code
├── src/                     # Source code modules
│   ├── download_data.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── run.py
│   └── visualize.py
├── app.py                   # Main Streamlit entry point
├── README.md                # You're here!
├── LICENSE
├── result.txt               # Model performance metrics
├── style.css                # Custom styling for app
└── environment.yml          # Conda environment dependencies
```

## Clone the repository
git clone https://github.com/pooja30123/Stock-Price-Prediction-using-LSTM.git
cd Stock-Price-Prediction-using-LSTM


## Set up environment using Conda
- conda env create -f environment.yml
- conda activate stock-predictor


## Run the app
streamlit run app.py


## 📧 Contact
If you have any questions or feedback, feel free to reach out!
