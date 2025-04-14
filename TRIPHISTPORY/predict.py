#Trip History Analysis: Use trip history dataset that is from a bike sharing service in the United States. The data is provided quarter-wise from 2010 (Q4) onwards. Each file has 7 columns. Predict the class of user.  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load and clean data
df = pd.read_csv(r"C:\Users\Manas\Desktop\com\2010-capitalbikeshare-tripdata.csv")
df.dropna(inplace=True)
df['Start date'] = pd.to_datetime(df['Start date'])

# Feature engineering
df['hour'] = df['Start date'].dt.hour
df['day'] = df['Start date'].dt.dayofweek
df['month'] = df['Start date'].dt.month
features = ['Duration', 'hour', 'day', 'month']
X = StandardScaler().fit_transform(df[features])
le = LabelEncoder()
y = le.fit_transform(df['Member type'])  # 0 = Customer, 1 = Subscriber

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("Recall   :", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1 Score :", round(f1_score(y_test, y_pred, average='weighted'), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Visualizations
sns.countplot(x='Member type', data=df)
plt.title('User Type Distribution')
plt.show()

sns.histplot(df['Duration'], bins=50, kde=True, color='skyblue')
plt.title('Trip Duration Distribution')
plt.show()

plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=features, class_names=le.classes_, filled=True)
plt.title('Decision Tree')
plt.show()
