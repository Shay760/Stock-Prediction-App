import streamlit as st
from datetime import date

import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Commands below create widgets for the web app
# gets the data from January 1st 2015 to today 
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
# gives the current date in the y-m-d format 

st.title("Stock Prediction App")

# Different stock options: Apple, Google, Microsoft, and Gamestop
stocks = ("AAPL", "GOOG", "MSFT", "GME")

# Select box for choosing between these stock options
selected_stock = st.selectbox("Select dataset for prediction", stocks)

# A slider to select the number of years for prediction 
n_years = st.slider("Years of prediction:", 1 , 4)
period = n_years * 365


# Loads the stock data and caches each stock option, so it doesn't have to download data again. 
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True) # Puts the date in the first column 
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

# Plot/analyse the data 
st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)