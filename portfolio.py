import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("📊 Portfolio Analyzer")
st.markdown("**By: Hemangi Suthar, Mahika Jain, Yashna Meher**")

# Fetch S&P 500 tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    tickers = tables[0]['Symbol'].tolist()
    return sorted(tickers)

# Sidebar Inputs
st.sidebar.header("Input Your Portfolio")
sp500_list = get_sp500_tickers()
portfolio_tickers = st.sidebar.multiselect(
    "Select stock tickers:",
    options=sp500_list,
    default=["AAPL", "MSFT", "GOOGL"],
    help="Search and select one or more S&P 500 stocks"
)

# Input investment amount per stock
st.sidebar.subheader("Enter amount invested in each stock ($)")
investment_input = {}
for ticker in portfolio_tickers:
    investment_input[ticker] = st.sidebar.number_input(f"Investment in {ticker}", min_value=0.0, value=100.0)

try:
    tickers = portfolio_tickers

    # Get latest prices
    latest_prices = {}
    for ticker in tickers:
        latest_prices[ticker] = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]

    # Calculate shares from dollar investment
    share_input = {ticker: investment_input[ticker] / latest_prices[ticker] for ticker in tickers}

    # Calculate holdings value
    holding_values = {ticker: latest_prices[ticker] * share_input[ticker] for ticker in tickers}
    total_value = sum(holding_values.values())
    weights = [holding_values[ticker] / total_value for ticker in tickers]

    # Data Download
    start_date, end_date = "2015-01-01", "2024-12-31"
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    # Extract Adjusted Close
    adj_close = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers})

    # Add date range selector with formatted display
    date_range = adj_close.index
    start_slider, end_slider = st.sidebar.select_slider(
        "Select Date Range for Analysis:",
        options=date_range,
        value=(date_range[0], date_range[-1]),
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )
    adj_close = adj_close.loc[start_slider:end_slider]

    returns = adj_close.pct_change().dropna()
    portfolio_returns = (returns * weights).sum(axis=1)
    portfolio_returns.name = "Portfolio_Return"

    # Market Returns
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)["Close"]
    sp500_returns = sp500.pct_change().dropna()
    aligned = pd.concat([portfolio_returns, sp500_returns], axis=1).dropna()
    aligned.columns = ["Portfolio_Return", "Market_Return"]

    # Cumulative Return Snapshot
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_yearly = portfolio_cumulative.resample("Y").last().pct_change().dropna()
    total_growth = (portfolio_cumulative[-1] - 1) * 100
    best_year = portfolio_yearly.idxmax().year
    best_return = portfolio_yearly.max() * 100
    worst_year = portfolio_yearly.idxmin().year
    worst_return = portfolio_yearly.min() * 100

    st.subheader("📅 Past Performance Snapshot")
    st.markdown(f"**Total Growth:** {total_growth:.2f}%")
    st.markdown(f"**Best Year:** {best_year} ({best_return:.2f}%)")
    st.markdown(f"**Worst Year:** {worst_year} ({worst_return:.2f}%)")

    # Metrics
    beta = np.round(np.polyfit(aligned["Market_Return"], aligned["Portfolio_Return"], 1)[0], 4)
    sharpe = np.round((portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252), 4)
    volatility = np.round(portfolio_returns.std() * np.sqrt(252), 4)
    var_95 = np.round(np.percentile(portfolio_returns, 5), 4)
    
    # GARCH Forecast
    garch_model = arch_model(portfolio_returns * 100, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    forecast = garch_fit.forecast(horizon=30)
    volatility_forecast = np.sqrt(forecast.variance.values[-1, :])

    st.subheader("📈 Cumulative Portfolio vs Market")
    cum_returns = (1 + aligned).cumprod()
    st.line_chart(cum_returns)

    st.subheader("📊 Portfolio Risk Metrics")
    st.markdown(f"**Beta:** {beta}")
    st.markdown(f"**Sharpe Ratio:** {sharpe}")
    st.markdown(f"**Annualized Volatility:** {volatility:.2%}")
    st.markdown(f"**Value at Risk (95% confidence):** {var_95:.2%}")

    st.subheader("🔮 Forecasted Volatility (Next 30 Days)")
    fig, ax = plt.subplots()
    ax.plot(volatility_forecast, marker='o')
    ax.set_title("GARCH Forecast (30-day Volatility)")
    ax.set_xlabel("Day Ahead")
    ax.set_ylabel("Volatility (%)")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("🧩 Sector Breakdown")
    sector_counts = {}
    for ticker in portfolio_tickers:
        try:
            sector = yf.Ticker(ticker).info.get("sector", "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        except:
            continue
    sector_df = pd.DataFrame.from_dict(sector_counts, orient="index", columns=["Count"])
    st.dataframe(sector_df)
    fig = px.pie(sector_df, values='Count', names=sector_df.index, title='Sector Allocation')
    st.plotly_chart(fig)

    st.subheader("🟢 Real-Time Prices")
    st.write(pd.DataFrame.from_dict(latest_prices, orient='index', columns=['Latest Price ($)']))

    # -------------------------------
    # 📊 Top Movers Today (Live %)
    # -------------------------------
    st.subheader("📊 Top Movers Today (Live %)")
    try:
        raw_data = yf.download(portfolio_tickers, period='2d', interval='1d', auto_adjust=True)
        if isinstance(raw_data.columns, pd.MultiIndex):
            adj_close = raw_data['Close'].T
        else:
            adj_close = raw_data[['Close']].rename(columns={'Close': portfolio_tickers[0]}).T
            adj_close.columns = raw_data.index.strftime("%Y-%m-%d")

        if adj_close.shape[1] < 2:
            st.warning("Not enough data to calculate % change.")
        else:
            prev_close = adj_close.iloc[:, 0]
            latest_close = adj_close.iloc[:, 1]
            change_pct = ((latest_close - prev_close) / prev_close * 100).sort_values(ascending=False)
            for ticker, pct in change_pct.head(3).items():
                arrow = "📈 Increased" if pct > 0 else "📉 Decreased"
                st.metric(label=f"{ticker}", value=f"{pct:.2f}%", delta=arrow)
    except Exception as e:
        st.warning(f"Top Movers data unavailable: {e}")

    # Monte Carlo Simulation
    st.subheader("🎲 Monte Carlo Simulation")
    num_days = 30
    num_simulations = 1000
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    simulated_paths = np.zeros((num_days, num_simulations))
    for i in range(num_simulations):
        daily_returns = np.random.normal(loc=mu, scale=sigma, size=num_days)
        price_path = np.cumprod(1 + daily_returns)
        simulated_paths[:, i] = total_value * price_path
    fig, ax = plt.subplots()
    ax.plot(simulated_paths, color="lightblue", alpha=0.3)
    ax.set_title("Monte Carlo Simulation of Portfolio Value")
    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value ($)")
    st.pyplot(fig)
    final_values = simulated_paths[-1, :]
    st.markdown(f"**Expected Mean Value:** ${np.mean(final_values):,.2f}")
    st.markdown(f"**95% VaR (loss limit):** ${total_value - np.percentile(final_values, 5):,.2f}")
    st.markdown(f"**Worst-case (simulated):** ${np.min(final_values):,.2f}")
    st.markdown(f"**Best-case (simulated):** ${np.max(final_values):,.2f}")

    # K-Means Clustering
    st.subheader("🧠 K-Means Clustering")
    features = pd.DataFrame({"Mean_Return": returns.mean(), "Volatility": returns.std()})
    if len(features) >= 2:
        n_clusters = min(3, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        features["Cluster"] = kmeans.fit_predict(features[["Mean_Return", "Volatility"]])
        fig = px.scatter(features, x="Volatility", y="Mean_Return", color="Cluster", text=features.index,
                         title="Clustering Assets by Risk-Return")
        st.plotly_chart(fig)

except Exception as e:
    st.error(f"Something went wrong: {e}")
