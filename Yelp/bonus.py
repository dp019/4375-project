import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the Yelp reviews dataset
df = pd.read_csv('Yelp Restaurant Reviews.csv')

# Drop neutral reviews (rating = 3)
df = df[df['Rating'] != 3]

# Create sentiment labels: 1 if rating >= 4, else 0
df['sentiment'] = df['Rating'].apply(lambda x: 1 if x >= 4 else 0)

# Clean the review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['cleaned_review'] = df['Review Text'].astype(str).apply(clean_text)

# Encode sentiment labels
y_encoded = df['sentiment'].values

# Parameters
max_words = 20000
max_len = 300
embedding_dim = 128

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['cleaned_review'])
X_seq = tokenizer.texts_to_sequences(df['cleaned_review'])
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_encoded, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
early_stop = EarlyStopping(patience=2, restore_best_weights=True)
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'LSTM Accuracy on Yelp Reviews: {accuracy:.4f}')

