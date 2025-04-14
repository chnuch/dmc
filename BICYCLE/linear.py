#Predict Bike Trips using Regression Models Use weather and time-related features to predict number of bicycle trips using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Bike Data
bike = pd.read_csv('D:\\Downloads\\Fremont_Bridge_Bicycle_Counter_20241227.csv')
bike.columns = ['Date', 'Total', 'West', 'East']
bike['Date'] = pd.to_datetime(bike['Date'])
bike = bike.set_index('Date').resample('D').sum()

# Load Weather Data
weather = pd.read_csv('D:\Downloads\seattle_weather_1948-2017.csv')
weather['DATE'] = pd.to_datetime(weather['DATE'])
weather = weather.set_index('DATE')

# Select useful weather columns: PRCP (precipitation), TMAX, TMIN
weather = weather[['PRCP', 'TMAX', 'TMIN']]
weather.columns = ['Precipitation', 'TempMax', 'TempMin']

# Merge bike and weather data
df = bike.merge(weather, left_index=True, right_index=True)

# Feature Engineering: Add temporal features
df['Weekday'] = df.index.weekday
df['Month'] = df.index.month
df['Year'] = df.index.year
df['IsWeekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Drop any NA rows
df.dropna(inplace=True)

# Define Features and Target
features = ['Precipitation', 'TempMax', 'TempMin', 'Weekday', 'Month', 'IsWeekend']
X = df[features]
y = df['Total']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluate Model
print("\n Linear Regression Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.2f}")

# Coefficients
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': lr.coef_})
print("\n Model Coefficients:")
print(coef_df)

# Plot: Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.xlabel('Actual Trips')
plt.ylabel('Predicted Trips')
plt.title('Actual vs Predicted Bicycle Trips')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid(True)
plt.show()
