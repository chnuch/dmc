#Predict the number of bicycle trips across Seattle's Fremont Bridge based on weather, season, and other factors and also Figure out what we can learn about people in Seattle from hourly commute data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('D:\\Downloads\\Fremont_Bridge_Bicycle_Counter_20241227.csv')

# Inspect column names
print("Columns in the dataset:", data.columns)

# Data Preprocessing
# Create a 'Total' column by summing the east and west sidewalk counts
data['Total'] = data['Fremont Bridge Sidewalks, south of N 34th St Cyclist West Sidewalk'] + \
                data['Fremont Bridge Sidewalks, south of N 34th St Cyclist East Sidewalk']

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data['Day'] = data['Date'].dt.dayofyear  # Day of the year
data['DayOfWeek'] = data['Date'].dt.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)
data['Month'] = data['Date'].dt.month  # Month of the year
data['Year'] = data['Date'].dt.year

# Lag feature for previous day's total count
data['Lag1'] = data['Total'].shift(1)

# Drop rows with NaN values (due to lagging)
data = data.dropna()

# Features and target
features = ['Day', 'DayOfWeek', 'Month', 'Lag1']
X = data[features]
y = data['Total']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a function to evaluate models with different configurations
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R-squared: {r2:.2f}')
    
    return y_pred

# Linear Regression Model
print("\nLinear Regression Model:")
lr_model = LinearRegression()
y_pred_lr = evaluate_model(lr_model, X_train, y_train, X_test, y_test)

# Ridge Regression Model (L2 regularization)
print("\nRidge Regression Model:")
ridge_model = Ridge(alpha=1.0)  # Regularization strength
y_pred_ridge = evaluate_model(ridge_model, X_train, y_train, X_test, y_test)

# Lasso Regression Model (L1 regularization)
print("\nLasso Regression Model:")
lasso_model = Lasso(alpha=0.1)  # Regularization strength
y_pred_lasso = evaluate_model(lasso_model, X_train, y_train, X_test, y_test)

# Hyperparameter Tuning: Grid Search for Ridge and Lasso
param_grid = {'alpha': [0.1, 1, 10, 100]}  # Regularization strength
ridge_search = GridSearchCV(Ridge(), param_grid, cv=5)
lasso_search = GridSearchCV(Lasso(), param_grid, cv=5)

# Train Ridge model with grid search
ridge_search.fit(X_train, y_train)
print(f"\nBest Ridge Alpha: {ridge_search.best_params_['alpha']}")
y_pred_ridge_search = ridge_search.predict(X_test)
print(f"Ridge - MSE: {mean_squared_error(y_test, y_pred_ridge_search):.2f}, R2: {r2_score(y_test, y_pred_ridge_search):.2f}")

# Train Lasso model with grid search
lasso_search.fit(X_train, y_train)
print(f"\nBest Lasso Alpha: {lasso_search.best_params_['alpha']}")
y_pred_lasso_search = lasso_search.predict(X_test)
print(f"Lasso - MSE: {mean_squared_error(y_test, y_pred_lasso_search):.2f}, R2: {r2_score(y_test, y_pred_lasso_search):.2f}")

# Predicting Future Values
future_days = pd.DataFrame({'Day': [366, 367, 368],  # Example future days
                            'DayOfWeek': [5, 6, 0],  # Example days of the week (Friday, Saturday, Sunday)
                            'Month': [12, 12, 12],  # December
                            'Lag1': [data['Total'].iloc[-1], data['Total'].iloc[-1], data['Total'].iloc[-1]]})

# Use the best model (e.g., Ridge with best alpha from grid search) for predictions
future_predictions = ridge_search.predict(future_days)

# Visualizing predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Total'], label='Actual Counts', color='blue')
plt.plot(data['Date'].iloc[-1] + pd.to_timedelta(np.arange(1, 4), 'D'), future_predictions, label='Predicted Counts', color='red')
plt.xlabel('Date')
plt.ylabel('Total Bicycle Count')
plt.title('Fremont Bicycle Counts - Linear Regression (Refined)')
plt.legend()
plt.show()

# Print Future Predictions
print("\nPredicted Bicycle Counts for Future Days:")
print(future_predictions)

# Scatter Plot: Total vs Day of Year
plt.figure(figsize=(10, 6))
plt.scatter(data['Day'], data['Total'], color='blue', alpha=0.5)
plt.title('Scatter Plot of Total Bicycle Count vs Day of Year')
plt.xlabel('Day of the Year')
plt.ylabel('Total Bicycle Count')
plt.show()

# Histogram: Distribution of Total Bicycle Counts
plt.figure(figsize=(10, 6))
plt.hist(data['Total'], bins=50, color='green', edgecolor='black')
plt.title('Histogram of Total Bicycle Counts')
plt.xlabel('Total Bicycle Count')
plt.ylabel('Frequency')
plt.show()
