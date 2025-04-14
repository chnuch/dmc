import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv("D:/Downloads/HousingData.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_raw = df.drop(columns=["MEDV"])
y = df["MEDV"]
X_array = imputer.fit_transform(X_raw)
X = pd.DataFrame(X_array, columns=X_raw.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Standardize features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Helper function to evaluate models ----
def evaluate_model(name, model, X_test, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{name} Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

# ---- 1. Linear Regression ----
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
evaluate_model("Linear Regression", lr, X_test_scaled, y_test, y_pred_lr)

# ---- 2. Ridge Regression ----
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
evaluate_model("Ridge Regression", ridge, X_test_scaled, y_test, y_pred_ridge)

# ---- 3. Lasso Regression ----
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
evaluate_model("Lasso Regression", lasso, X_test_scaled, y_test, y_pred_lasso)

# ---- 4. Polynomial Regression (degree 2) ----
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()
poly_lr.fit(X_train_poly, y_train)
y_pred_poly = poly_lr.predict(X_test_poly)
evaluate_model("Polynomial Regression (deg=2)", poly_lr, X_test_poly, y_test, y_pred_poly)

# ---- Plot Actual vs Predicted for best model ----
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_poly)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted - Polynomial Regression (deg=2)")
plt.tight_layout()
plt.show()
