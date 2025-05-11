import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load your cleaned Amazon reviews
df = pd.read_csv('cleaned_amazon_reviews.csv')

# (Optional) If your 'text' column still needs minimal cleaning:
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)               # strip HTML
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)                 # strip digits
    text = re.sub(r'\s+', ' ', text).strip()        # collapse whitespace
    return text

df['cleaned_text'] = df['text'].astype(str).apply(clean_text)

# 2. Prepare inputs and labels
X = df['cleaned_text'].values
y = df['sentiment'].values

# 3. Tokenize & pad
max_words     = 20000
max_len       = 300
embedding_dim = 128

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_pad     = pad_sequences(sequences, maxlen=max_len)

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# 5. Build the LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 6. Train with early stopping
early_stop = EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 7. Evaluate on held-out test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'LSTM Accuracy on Amazon Reviews: {accuracy:.4f}')
