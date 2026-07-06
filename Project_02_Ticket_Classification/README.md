# Customer Support Ticket Classification & Prioritization

This repository contains my submission for the NLP Ticket Classification assessment. The objective is to build a machine learning model that categorizes customer support tickets and predicts their priority level based on the ticket description.

I chose to implement this using a deep learning approach (LSTM) rather than traditional bag-of-words methods to better capture the context of the text.

## Files Included
- `customer_support_tickets.csv`: The raw dataset containing ticket descriptions, categories, and priority levels.
- `Ticket_Classification_Colab.ipynb`: A Jupyter Notebook intended to be run in Google Colab. It demonstrates the full Deep Learning pipeline.
- `ticket_classification_colab.py`: A standalone Python script that performs the exact same pipeline if you prefer running it locally via terminal.

## Methodology (LSTM)
I used a Long Short-Term Memory (LSTM) Neural Network built with TensorFlow/Keras. LSTMs are useful for text processing because they read sequences sequentially, meaning the model can pick up on word order and context rather than just counting word frequencies like a TF-IDF approach would.

The pipeline performs the following steps:
1. **Text Preprocessing**: Cleans the raw text (lowercasing, punctuation removal) and converts the string categories into numerical labels.
2. **Tokenization & Padding**: Converts the text into sequences of integers and pads them so they can be fed into the neural network in uniform batches.
3. **Model 1 (Category)**: Trains an LSTM to predict the `Ticket Type` (e.g., Billing, Technical issue).
4. **Model 2 (Priority)**: Trains an LSTM to predict the `Ticket Priority` (e.g., High, Low).
5. **Evaluation**: Generates a full classification report (Accuracy, Precision, Recall) and plots Confusion Matrices for both models to visualize exactly where the models succeed.
