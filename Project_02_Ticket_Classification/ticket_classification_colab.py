# Customer Support Ticket Classification
# LSTM neural network architecture

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import warnings

# 1. Load and Prepare Data
df = pd.read_csv('customer_support_tickets.csv')
df = df[['Ticket Description', 'Ticket Type', 'Ticket Priority']].dropna()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['clean_desc'] = df['Ticket Description'].apply(clean_text)

# 2. Text Preprocessing (Tokenization and Padding)
type_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

df['Type_Encoded'] = type_encoder.fit_transform(df['Ticket Type'])
df['Priority_Encoded'] = priority_encoder.fit_transform(df['Ticket Priority'])

MAX_WORDS = 10000
MAX_LEN = 100

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_desc'])
sequences = tokenizer.texts_to_sequences(df['clean_desc'])
X_padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Shape of padded text data: {X_padded.shape}")

# 3. Build Neural Network Architecture
def build_lstm_model(vocab_size, max_length, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length, mask_zero=True),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
