#Download Boston Housing dataset. Create a Model using linear regression to predict the houses price. 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("D:/Downloads/HousingData.csv")

X_raw = df.drop(columns=["MEDV"])
y = df["MEDV"]

imputer = SimpleImputer(strategy="mean")
X_array = imputer.fit_transform(X_raw)
X = pd.DataFrame(X_array, columns=X_raw.columns)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X["RM"], y=y, label="Actual")
rm_model = LinearRegression()
rm_model.fit(X[["RM"]], y)
rm_line = rm_model.predict(X[["RM"]])
plt.plot(X["RM"], rm_line, color="red", label="Regression Line")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median Value of Homes (MEDV)")
plt.title("Linear Regression: RM vs MEDV")
plt.legend()
plt.tight_layout()
plt.show()
