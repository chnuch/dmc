#Predict User Class (Subscriber vs Customer) : Build a classification model (like Logistic Regression, Random Forest, or XGBoost) to predict user class using engineered features. Evaluate using accuracy, precision, recall, and F1 score.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load your dataset
df = pd.read_csv(r"C:\Users\Manas\Desktop\com\2010-capitalbikeshare-tripdata.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Convert 'Start date' to datetime
df['Start date'] = pd.to_datetime(df['Start date'])

# Feature Engineering: Extract time-based features
df['hour'] = df['Start date'].dt.hour
df['day'] = df['Start date'].dt.dayofweek
df['month'] = df['Start date'].dt.month

# Select features
features = ['Duration', 'hour', 'day', 'month']
X = df[features]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target variable ('Member type')
le = LabelEncoder()
y = le.fit_transform(df['Member type'])  # 0 = Customer, 1 = Subscriber

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n Model: {name}")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    
    # Using average='weighted' for multiclass classification
    print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, average='weighted'), 4))
    print("F1 Score :", round(f1_score(y_test, y_pred, average='weighted'), 4))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
