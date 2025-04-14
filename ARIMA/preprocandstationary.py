#Preprocessing and Stationarity Check: Load the COVID-19 dataset from Kaggle. Clean and analyze the time series. Apply stationarity tests (ADF) and visualize trends and seasonality.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
df = pd.read_csv(r"D:\Downloads\StatewiseTestingDetails (1).csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.groupby('Date')['Positive'].sum().asfreq('D').fillna(method='ffill')

# Decompose the time series
decomposition = seasonal_decompose(df, model='additive')
decomposition.trend.plot(title='Trend')
plt.show()
decomposition.seasonal.plot(title='Seasonality')
plt.show()
decomposition.resid.plot(title='Residuals')
plt.show()

# ADF Test
result = adfuller(df)

print('ADF Statistic:', result[0])
print('p-value:', result[1])
if result[1] > 0.05:
    print("The series is NOT stationary. Differencing is required.")
else:
    print("The series is stationary.")
