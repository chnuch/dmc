import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\Manas\Documents\Book1.csv")

# Step 2: Split data into input (X) and output (y)
X = df[['hours']]  # Features (must be 2D)
y = df['scores']    # Target

# Step 3: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 4: Print the model equation
slope = model.coef_[0]
intercept = model.intercept_
print(f"Model: Score = {slope:.2f} * Hours + {intercept:.2f}")

# Step 5: Make predictions
df['Predicted Score'] = model.predict(X)
print(df)

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\Manas\Documents\Book1.csv")

# Step 2: Split data into input (X) and output (y)
X = df[['hours']]  # Features (must be 2D)
y = df['scores']    # Target

# Step 3: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 4: Print the model equation
slope = model.coef_[0]
intercept = model.intercept_
print(f"Model: Score = {slope:.2f} * Hours + {intercept:.2f}")

# Step 5: Make predictions
df['Predicted Score'] = model.predict(X)
print(df)

plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, df['Predicted Score'], color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.legend()
plt.show()
