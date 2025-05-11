import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Yelp reviews CSV
df = pd.read_csv("Yelp Restaurant Reviews.csv")

# Rename columns for consistency
df = df.rename(columns={"Review Text": "review", "Rating": "rating"})

# Create sentiment labels
df["sentiment"] = df["rating"].apply(lambda x: "positive" if x >= 4 else "negative")

# Drop rows with missing reviews
df = df.dropna(subset=["review"])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML
    text = re.sub(r"[^a-z\s]", "", text)  # Remove punctuation/numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# Apply preprocessing
df["cleaned_review"] = df["review"].apply(preprocess_text)

# Split data
X = df["cleaned_review"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# SVM
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_vec, y_train)
log_preds = log_model.predict(X_test_vec)

# Print results
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))
print("\nClassification Report for SVM:\n", classification_report(y_test, svm_preds))
