import matplotlib.pyplot as plt

# Data
X = [1, 2, 3, 4, 5]
Y = [52, 56, 61, 65, 71]
n = len(X)

# Step 1: Calculate means
mean_x = sum(X) / n
mean_y = sum(Y) / n

# Step 2: Calculate numerator and denominator for slope
numerator = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
denominator = sum((X[i] - mean_x)**2 for i in range(n))

# Step 3: Calculate slope and intercept
m = numerator / denominator
b = mean_y - m * mean_x

print(f"Linear Regression Equation: Score = {m:.2f} * Hours + {b:.2f}")

# Step 4: Predict values
print("\nPredictions:")
for i in range(n):
    predicted = m * X[i] + b
    print(f"Hours: {X[i]}, Actual Score: {Y[i]}, Predicted: {predicted:.2f}")

# Step 5: Prepare predicted values for the line
predicted_Y = [m * x + b for x in X]

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', label='Actual Scores')
plt.plot(X, predicted_Y, color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Linear Regression: Hours vs Score')
plt.legend()
plt.grid(True)
plt.show()
