import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer # pyright: ignore[reportMissingImports]

# Config
CSV_PATH = "train.csv"
MAX_WORDS = 20000
MAX_LEN = 100   # must match app.py MAX_LENGTH
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 3      # keep small for quick training
MODEL_OUT = "toxicity.h5"
TOKENIZER_OUT = "tokenizer.pkl"

# Load data
df = pd.read_csv(CSV_PATH)
# detect text column
text_cols = [c for c in df.columns if "comment" in c.lower() or "text" in c.lower()]
if not text_cols:
    raise SystemExit("No text/comment column found in train.csv. Rename column to include 'comment' or 'text'.")
text_col = text_cols[0]

# detect common label columns
possible_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
label_cols = [c for c in df.columns if c in possible_labels]
if not label_cols:
    # fallback: any binary columns except text
    candidates = [c for c in df.columns if c!=text_col and df[c].dropna().isin([0,1]).all()]
    if not candidates:
        raise SystemExit("No label columns found. Ensure dataset has binary label columns.")
    label_cols = candidates

print("Using text column:", text_col)
print("Detected label columns:", label_cols)

# Prepare texts and labels
texts = df[text_col].fillna("").astype(str).values
labels = df[label_cols].fillna(0).astype(float).values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels[:,0] if labels.shape[1]>0 else None)

# Tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
with open(TOKENIZER_OUT, "wb") as f:
    pickle.dump(tokenizer, f)
print("Saved tokenizer to", TOKENIZER_OUT)

# Sequences
seq_train = tokenizer.texts_to_sequences(X_train)
seq_val = tokenizer.texts_to_sequences(X_val)
Xtr = pad_sequences(seq_train, maxlen=MAX_LEN)
Xv = pad_sequences(seq_val, maxlen=MAX_LEN)

# Build simple model
num_labels = y_train.shape[1]
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_labels, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["AUC"])
model.summary()

# Callbacks
es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
ck = ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_loss")

# Train
history = model.fit(
    Xtr, y_train,
    validation_data=(Xv, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, ck]
)

# Save final model (ModelCheckpoint already saved best)
model.save(MODEL_OUT)
print("Saved model to", MODEL_OUT)