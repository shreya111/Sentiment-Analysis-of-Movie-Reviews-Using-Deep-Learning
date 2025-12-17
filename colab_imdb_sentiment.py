#!/usr/bin/env python3
"""
Sentiment Analysis of Movie Reviews Using Deep Learning (Colab-ready)

This script trains a Recurrent Neural Network (RNN - BiLSTM) to classify
IMDB movie reviews as positive or negative. It is designed to run end-to-end
in Google Colab with minimal setup.

Highlights
- Data source preference: Kaggle IMDB 50K Movie Reviews (if Kaggle API is available)
- Automatic fallback to TensorFlow Datasets (imdb_reviews) if Kaggle credentials are not provided
- Clean, tokenize, and pad sequences
- Train a BiLSTM with embeddings
- Evaluate with accuracy and classification report
- Show sample predictions at the end

How to run in Google Colab
1) Open a new Colab notebook: https://colab.research.google.com
2) Upload this file to your Colab session (Files panel → Upload)
3) Run: !python colab_imdb_sentiment.py

Optional: Use Kaggle dataset (preferred)
- In Colab, add your Kaggle API credentials (kaggle.json) as described here:
  https://www.kaggle.com/docs/api
- Then run the script; it will detect credentials automatically.

"""

import os
import sys
import re
import string
import random
import json
from pathlib import Path
from typing import Tuple, List

# Step 0 — Minimal installs for Colab environments
# We install kaggle + tensorflow-datasets quietly. TensorFlow usually comes preinstalled in Colab.
# If you run outside Colab, these installs are still safe.
print("Step 0: Installing/validating dependencies (this may take a minute the first time)...")
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# Use pip programmatically to ensure required packages are available.
# We keep TensorFlow as the default included in Colab; scikit-learn and pandas also usually exist,
# but we ensure they're present.

def pip_install(packages: List[str]):
    import subprocess
    for pkg in packages:
        try:
            __import__(pkg.split("==")[0].replace("-", "_"))
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

pip_install([
    "kaggle",
    "pandas",
    "scikit-learn",
    "tensorflow-datasets",
])

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, callbacks

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print("GPU available:", bool(tf.config.list_physical_devices('GPU')))

# Paths
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "imdb_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Step 1 — Load data
# Priority: Kaggle IMDB 50K dataset → TFDS imdb_reviews fallback
print("\nStep 1: Loading data (Kaggle if available, otherwise TFDS)...")

KAGGLE_DATASET_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"  # contains IMDB Dataset.csv
KAGGLE_DATA_FILE = DATA_DIR / "IMDB Dataset.csv"

def has_kaggle_credentials() -> bool:
    # Either kaggle.json in ~/.kaggle or KAGGLE_USERNAME/KEY environment variables
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists() or ("KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ)


def download_kaggle_dataset() -> bool:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print("Kaggle API not available:", e)
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        print(f"Downloading Kaggle dataset: {KAGGLE_DATASET_SLUG} ...")
        api.dataset_download_files(KAGGLE_DATASET_SLUG, path=str(DATA_DIR), unzip=True)
        if KAGGLE_DATA_FILE.exists():
            print("Kaggle dataset downloaded successfully.")
            return True
        else:
            print("Kaggle dataset file not found after download.")
            return False
    except Exception as e:
        print("Failed to download from Kaggle:", e)
        return False


def load_from_kaggle() -> pd.DataFrame:
    if not KAGGLE_DATA_FILE.exists():
        raise FileNotFoundError("Kaggle data file not found.")
    df = pd.read_csv(KAGGLE_DATA_FILE)
    # Expected columns: 'review', 'sentiment' (positive/negative)
    df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)
    return df


def load_from_tfds() -> pd.DataFrame:
    import tensorflow_datasets as tfds
    print("Loading 'imdb_reviews' from TensorFlow Datasets...")
    ds_train = tfds.load("imdb_reviews", split="train", as_supervised=True)
    ds_test = tfds.load("imdb_reviews", split="test", as_supervised=True)

    def ds_to_lists(ds):
        texts, labels = [], []
        for text, label in tfds.as_numpy(ds):
            texts.append(text.decode("utf-8"))
            labels.append(int(label))
        return texts, labels

    x_train, y_train = ds_to_lists(ds_train)
    x_test, y_test = ds_to_lists(ds_test)
    df_train = pd.DataFrame({"review": x_train, "label": y_train})
    df_test = pd.DataFrame({"review": x_test, "label": y_test})
    df = pd.concat([df_train, df_test], ignore_index=True)
    # Map numeric to sentiment string for consistency
    df["sentiment"] = df["label"].map({1: "positive", 0: "negative"})
    df = df.drop(columns=["label"]).reset_index(drop=True)
    return df

USE_KAGGLE = False
if has_kaggle_credentials():
    if download_kaggle_dataset():
        USE_KAGGLE = True
    else:
        print("Falling back to TensorFlow Datasets (imdb_reviews)...")
else:
    print("Kaggle credentials not found. Falling back to TensorFlow Datasets (imdb_reviews)...")

if USE_KAGGLE:
    df_all = load_from_kaggle()
else:
    df_all = load_from_tfds()

print("Loaded records:", len(df_all))
print("Class balance:\n", df_all["sentiment"].value_counts())

# Step 2 — Preprocessing and cleaning
print("\nStep 2: Cleaning and preprocessing text...")

HTML_TAG_RE = re.compile(r"<.*?>")
URL_RE = re.compile(r"http\S+|www\S+")

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(HTML_TAG_RE, " ", s)
    s = re.sub(URL_RE, " ", s)
    # Replace common HTML entities
    s = s.replace("&amp;", " and ").replace("&quot;", " ")
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Apply cleaning
df_all["clean_review"] = df_all["review"].astype(str).apply(clean_text)
df_all["label"] = (df_all["sentiment"].str.lower() == "positive").astype(int)

# Train / test split
train_df, test_df = train_test_split(
    df_all[["clean_review", "label"]],
    test_size=0.2,
    random_state=SEED,
    stratify=df_all["label"],
)

x_train = train_df["clean_review"].tolist()
y_train = train_df["label"].tolist()
x_test = test_df["clean_review"].tolist()
y_test = test_df["label"].tolist()

print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")

# Tokenize and pad
VOCAB_SIZE = 20000
MAX_LEN = 200
OOV_TOKEN = "<OOV>"

print("Fitting tokenizer...")
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(x_train)

print("Converting texts to padded sequences...")
X_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LEN, padding="post", truncating="post")
X_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LEN, padding="post", truncating="post")
Y_train = np.array(y_train)
Y_test = np.array(y_test)

# Step 3 — Build the RNN model
print("\nStep 3: Building the BiLSTM model...")
EMBED_DIM = 128
LSTM_UNITS = 128
DROPOUT = 0.3

model = models.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
    layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=False)),
    layers.Dropout(DROPOUT),
    layers.Dense(64, activation="relu"),
    layers.Dropout(DROPOUT),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)

model.summary()

# Step 4 — Train
print("\nStep 4: Training...")
BATCH_SIZE = 128
EPOCHS = 3  # Increase if you have more time/compute

es = callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1)

history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, rlr],
    verbose=1,
)

# Step 5 — Evaluate
print("\nStep 5: Evaluating on test set...")
probs = model.predict(X_test, batch_size=256, verbose=0).ravel()
preds = (probs >= 0.5).astype(int)
acc = accuracy_score(Y_test, preds)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(Y_test, preds, target_names=["negative", "positive"]))
print("Confusion Matrix:\n", confusion_matrix(Y_test, preds))

# Step 6 — Try some custom sentences
print("\nStep 6: Sample predictions...")
examples = [
    "What a fantastic movie! The performances were stellar and I loved every minute.",
    "Absolutely terrible. Boring plot, wooden acting, and a complete waste of time.",
    "It had some good moments but overall it was just okay.",
    "A masterpiece that kept me engaged from start to finish.",
    "I didn't enjoy it at all, the pacing was off and the script was weak.",
]

ex_clean = [clean_text(t) for t in examples]
ex_seq = pad_sequences(tokenizer.texts_to_sequences(ex_clean), maxlen=MAX_LEN, padding="post", truncating="post")
ex_prob = model.predict(ex_seq, verbose=0).ravel()

for text, p in zip(examples, ex_prob):
    label = "positive" if p >= 0.5 else "negative"
    print(f"- {label:8s} (p={p:.3f}): {text}")

# Optional: Save artifacts
MODEL_DIR = BASE_DIR / "imdb_model"
MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_DIR / "bilstm_imdb.h5")

with open(MODEL_DIR / "tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())

print("\nAll done! Model and tokenizer saved to:", MODEL_DIR)
