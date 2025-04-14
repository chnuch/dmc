#Consider a suitable text dataset. Remove stop words, apply stemming and feature selection techniques to represent documents as vectors. Classify documents and evaluate precision, recall. 

import nltk
import string
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization + stopword removal
    return " ".join(words)

# Load dataset
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']  # Select categories
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
X_raw, y = newsgroups.data, newsgroups.target

# Apply preprocessing
X_processed = [preprocess(text) for text in X_raw]

# Convert text to TF-IDF vectors (including bigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use unigrams + bigrams
X = vectorizer.fit_transform(X_processed)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Naïve Bayes (Optimizing alpha)
param_grid = {'alpha': np.linspace(0.1, 1.0, 10)}  # Try different alpha values
model = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
model.fit(X_train, y_train)

# Best model selection
best_nb = model.best_estimator_

# Make Predictions
y_pred = best_nb.predict(X_test)

# Evaluate Precision, Recall, F1-score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print Results
print(f"Best Alpha: {model.best_params_['alpha']}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
