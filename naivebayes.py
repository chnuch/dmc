#Train a classifier Naive Bayes on the vectorized dataset. Evaluate the classifier using precision, recall, and F1 score. Perform cross-validation for robustness.

import nltk
import string
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(words)

# Load dataset
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
X_raw, y = data.data, data.target
target_names = data.target_names

# Preprocess documents
print("Preprocessing documents...")
X_processed = [preprocess(doc) for doc in X_raw]

# TF-IDF vectorization (unigrams + bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
X = vectorizer.fit_transform(X_processed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train model
print("Training model...")
nb_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = nb_classifier.predict(X_test)

# Evaluate with precision, recall, F1
print("\nTest Set Evaluation:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Cross-validation for robustness
print("Performing 5-fold cross-validation...")
cv_precision = cross_val_score(nb_classifier, X, y, cv=5, scoring='precision_macro')
cv_recall = cross_val_score(nb_classifier, X, y, cv=5, scoring='recall_macro')
cv_f1 = cross_val_score(nb_classifier, X, y, cv=5, scoring='f1_macro')

print("\nCross-Validation Scores (5-fold):")
print(f"Average Precision: {cv_precision.mean():.4f}")
print(f"Average Recall: {cv_recall.mean():.4f}")
print(f"Average F1 Score: {cv_f1.mean():.4f}")

# Confusion Matrix
y_cv_pred = cross_val_predict(nb_classifier, X, y, cv=5)
cm = confusion_matrix(y, y_cv_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Cross-Validated Predictions)")
plt.tight_layout()
plt.show()
