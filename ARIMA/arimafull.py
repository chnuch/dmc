import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv(r"D:\Downloads\StatewiseTestingDetails (1).csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.groupby('Date')['Positive'].sum().asfreq('D').fillna(method='ffill')

decomposition = seasonal_decompose(df, model='additive')
decomposition.trend.plot(title='Trend')
plt.show()
decomposition.seasonal.plot(title='Seasonality')
plt.show()
decomposition.resid.plot(title='Residuals')
plt.show()

result = adfuller(df)
print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] > 0.05:
    print("The series is NOT stationary. Differencing is required.")
    df_diff = df.diff().dropna()
else:
    print("The series is stationary.")
    df_diff = df

model = ARIMA(df, order=(5, 1, 0)) 
model_fit = model.fit()

forecast = model_fit.forecast(steps=30)

print(model_fit.summary())

plt.figure(figsize=(12, 6))
plt.plot(df, label='Original')
plt.plot(forecast.index, forecast, label='Forecast (Next 30 Days)', color='red')
plt.title('COVID-19 Positive Cases Forecast')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.legend()
plt.grid(True)
plt.show()
