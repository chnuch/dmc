#Exploratory Data Analysis (EDA) on Bicycle Data Perform EDA on hourly/daily bike count data. Analyze seasonal trends, weather impact, day vs night usage, and holiday effects.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar

# Load the Fremont dataset
df = pd.read_csv("D:\\Downloads\\Fremont_Bridge_Bicycle_Counter_20241227.csv")

# Rename columns for clarity
df.columns = ['Date', 'Total', 'West', 'East']

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set datetime index
df.set_index('Date', inplace=True)

# If data is hourly, resample to daily
daily = df.resample('D').sum()

# Add date-related features
daily['Weekday'] = daily.index.weekday
daily['Month'] = daily.index.month
daily['Year'] = daily.index.year
daily['Day'] = daily.index.day
daily['IsWeekend'] = daily['Weekday'].apply(lambda x: x >= 5)

# ----------------------------- SEASONAL TRENDS -----------------------------
plt.figure(figsize=(10, 5))
daily.groupby('Year')['Total'].sum().plot(kind='bar', color='skyblue')
plt.title("Total Bike Counts per Year")
plt.ylabel("Total Count")
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 5))
daily.groupby('Month')['Total'].mean().plot(kind='bar', color='salmon')
plt.title("Average Bike Count by Month")
plt.ylabel("Average Daily Count")
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(data=daily, x='Weekday', y='Total')
plt.title("Weekly Bike Count Patterns")
plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()

# ----------------------------- DAY vs NIGHT USAGE -----------------------------
df_hourly = pd.read_csv("D:\\Downloads\\Fremont_Bridge_Bicycle_Counter_20241227.csv")
df_hourly.columns = ['Date', 'Total', 'West', 'East']
df_hourly['Date'] = pd.to_datetime(df_hourly['Date'])
df_hourly['Hour'] = df_hourly['Date'].dt.hour
df_hourly['DayTime'] = df_hourly['Hour'].apply(lambda h: 'Day' if 7 <= h < 19 else 'Night')

usage = df_hourly.groupby('DayTime')['Total'].sum()

plt.figure(figsize=(6, 6))
usage.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
plt.title("Day vs Night Bicycle Usage")
plt.ylabel('')
plt.show()

# ----------------------------- HOLIDAY EFFECT -----------------------------
calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(start=daily.index.min(), end=daily.index.max())

daily['IsHoliday'] = daily.index.isin(holidays)

plt.figure(figsize=(8, 5))
sns.boxplot(data=daily, x='IsHoliday', y='Total')
plt.xticks([0, 1], ['Non-Holiday', 'Holiday'])
plt.title("Bike Usage: Holiday vs Non-Holiday")
plt.show()

# ----------------------------- WEATHER IMPACT -----------------------------
weather = pd.read_csv("D:\Downloads\seattle_weather_1948-2017.csv", parse_dates=['DATE'])
weather.set_index('DATE', inplace=True)
weather = weather[['PRCP', 'TMAX', 'TMIN']]

daily_weather = daily.merge(weather, left_index=True, right_index=True)

plt.figure(figsize=(10, 5))
sns.scatterplot(data=daily_weather, x='PRCP', y='Total', alpha=0.5)
plt.title("Bicycle Count vs Precipitation")
plt.xlabel("Precipitation (inches)")
plt.ylabel("Total Bicycle Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=daily_weather, x='TMAX', y='Total', alpha=0.5)
plt.title("Bicycle Count vs Max Temperature")
plt.xlabel("Max Temperature (°F)")
plt.ylabel("Total Bicycle Count")
plt.show()
