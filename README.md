# 📊 Portfolio Risk Analyzer: Real-Time Financial Analytics Dashboard

> Built by **Mahika Jain**, **Hemangi Suthar** and **Yashna Meher**

🔗 **Live App:** [portfolioanalyzerfinancial.streamlit.app](https://portfolioanalyzerfinancial.streamlit.app/)

---

## 📌 Overview

This project is an end-to-end financial analytics solution that empowers users to analyze and forecast the performance of stock portfolios using **real-time data** and **dynamic machine learning models**.

It consists of:
- A **Colab notebook** for initial data exploration and LSTM model development.
- A **Streamlit dashboard** for interactive, real-time portfolio analysis and forecasting.

---

## 🧪 Initial Colab Development

The Colab notebook laid the foundation for:
- Cleaning and preprocessing historical stock data.
- Visualizing stock trends and performance.
- Training an **LSTM neural network** to predict stock closing prices using a sliding-window approach.

### 💡 Portfolio Assumptions (in Notebook)
To simplify validation and logic development:
- Portfolio returns were calculated using **equal weights** for all selected stocks.
- This approach enabled:
  - Consistent performance comparisons
  - Easier debugging and testing of return/risk metrics
  - A clean base for extending logic later in the Streamlit app

---

## ⚡ Streamlit Application Highlights

The deployed Streamlit app adds a **real-world layer of interactivity and real-time data**. Key features include:

### 🔄 Real-Time Data via yfinance
- Stock data is fetched with **1-minute intervals** using the `yfinance` API.
- After fetching live prices, the app calculates **real-time portfolio return** by comparing latest and previous close values.

### 🧮 Investment-Aware Portfolio Logic
- Users can input the **actual amount invested** in each stock.
- The app dynamically computes **portfolio weights** based on these investments, giving personalized and accurate insights.

### 📈 Dynamic Forecasting Models
Unlike offline models, this app builds forecasting logic on the fly using live data:

- **GARCH**: Estimates market volatility
- **Monte Carlo Simulation**: Simulates possible future price movements
- **K-Means Clustering**: Identifies similar behavior across stocks
- **LSTM** (trained in Colab): Predicts future stock prices based on past sequences

---

## 🧰 Tech Stack

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Statsmodels
- **Visualization**: Matplotlib, Seaborn, Plotly
- **APIs**: yfinance
- **Framework**: Streamlit

---
