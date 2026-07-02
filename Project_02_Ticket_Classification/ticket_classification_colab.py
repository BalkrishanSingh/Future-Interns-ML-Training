# Customer Support Ticket Classification
# Data preparation and label encoding

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
