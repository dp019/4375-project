import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression



# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
# from sklearn.preprocessing import LabelEncoder

# Load CSV file
df = pd.read_csv('IMDB Dataset.csv')


# PART 1 OF PROJECT: DATA PREPROCESSING

# Define a simple preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]  # remove stopwords
    return ' '.join(tokens)

# Apply preprocessing to all reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

# # Show some sample output
print(df[['review', 'cleaned_review', 'sentiment']].head())


# PART 2 OF PROJECT: CLASSIFICATION


# 1. Split data
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Convert text to vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# 4. Train SVM
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_vec, y_train)
log_preds = log_model.predict(X_test_vec)

# 6. Evaluate Logistic Regression



# 5. Evaluate both
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_preds))
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_preds))
print("\nClassification Report for SVM:\n", classification_report(y_test, svm_preds))



df[['cleaned_review', 'sentiment']].to_csv('cleaned_imdb_reviews.csv', index=False)

