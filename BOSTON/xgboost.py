import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean dataset
df = pd.read_csv(r"C:\Users\Manas\Desktop\com\2010-capitalbikeshare-tripdata.csv")
df.dropna(inplace=True)
df['Start date'] = pd.to_datetime(df['Start date'])

# Feature engineering
df['hour'] = df['Start date'].dt.hour
df['day'] = df['Start date'].dt.dayofweek
df['month'] = df['Start date'].dt.month

features = ['Duration', 'hour', 'day', 'month']
X = df[features]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['Member type'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Evaluation
print("XGBoost\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# XGBoost ROC Curve
y_prob = xgb.predict_proba(X_test)[:, 1]  # Probabilities for class '1' (Subscriber)
fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='XGBoost (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.grid(True)
plt.show()


# Feature Importance
plot_importance(xgb)
plt.title("Feature Importance - XGBoost")
plt.show()
