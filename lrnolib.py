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
