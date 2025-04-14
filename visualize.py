#ARIMA Modeling for Forecasting : Use ARIMA models to forecast future COVID cases. Visualize predictions.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv(r"D:\Downloads\StatewiseTestingDetails (1).csv")
df['Date'] = pd.to_datetime(df['Date'])

# Prepare time series
ts = df.groupby('Date')['Positive'].sum().asfreq('D').fillna(method='ffill')

# Fit ARIMA model (example order)
model = ARIMA(ts, order=(5, 1, 0))
model_fit = model.fit()

# Forecast next 30 days
forecast = model_fit.forecast(steps=30)

# Plot historical data and forecast
plt.figure(figsize=(10, 4))
plt.plot(ts, label='Observed')
plt.plot(forecast.index, forecast, label='Forecast', color='red')
plt.title('COVID-19 Positive Cases Forecast (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Positive Cases')
plt.legend()
plt.show()
