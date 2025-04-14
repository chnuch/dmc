#Data Preprocessing for Trip History Load the bike-sharing dataset. Clean and preprocess data by handling missing values, date-time parsing, and feature engineering (e.g., duration bins, user age groups).

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = r"C:\Users\Manas\Desktop\com\2010-capitalbikeshare-tripdata.csv"

bike_data = pd.read_csv(file_path)

# Check the column names in your dataset
print(bike_data.columns)

# Handle Missing Values
missing_values = bike_data.isnull().sum()
print(f"Missing Values:\n{missing_values}\n")

# Drop rows with missing 'Start date' or 'End date' as they are essential for time calculations
bike_data = bike_data.dropna(subset=['Start date', 'End date'])

# Convert 'Start date' and 'End date' to datetime format
bike_data['Start date'] = pd.to_datetime(bike_data['Start date'])
bike_data['End date'] = pd.to_datetime(bike_data['End date'])

# Calculate Trip Duration (in seconds)
bike_data['trip_duration'] = (bike_data['End date'] - bike_data['Start date']).dt.total_seconds()

# Bin Trip Duration into Categories
duration_bins = [0, 600, 1800, 3600, float('inf')]  # 0-10 min, 10-30 min, 30-60 min, >60 min
duration_labels = ['short', 'medium', 'long', 'very_long']
bike_data['duration_group'] = pd.cut(bike_data['trip_duration'], bins=duration_bins, labels=duration_labels)

# Normalize the Trip Duration
scaler = MinMaxScaler()
bike_data['normalized_trip_duration'] = scaler.fit_transform(bike_data[['trip_duration']])

# Print the first few rows of processed data
print(f"Processed Data:\n{bike_data.head()}")
