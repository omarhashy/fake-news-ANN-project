# Fake News Detection Project

## Introduction to RNNs and NLP Concepts

### Recurrent Neural Networks (RNNs) and LSTMs

Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to recognize patterns in sequences of data, such as text, audio, or time series. Unlike traditional neural networks that process inputs independently, RNNs maintain an internal memory to process sequences of inputs. They have connections that form directed cycles, allowing the output from a previous step to influence the current step's input.

In this project, we use a specialized type of RNN called Long Short-Term Memory (LSTM) networks. LSTMs address the "vanishing gradient problem" that affects standard RNNs by incorporating a memory cell that can maintain information for long periods. This is particularly useful for text analysis, as it allows the model to remember important context from earlier parts of an article when making classification decisions.

Furthermore, we use Bidirectional LSTMs, which process data in both forward and backward directions, allowing the network to capture context from both past and future words in a sequence. This bidirectional approach significantly improves the model's ability to understand the nuances and context in news articles, enhancing fake news detection accuracy.

### Text Processing and Tokenization

Before feeding text into neural networks, we need to convert it into a numerical form. This conversion process involves several key concepts:

- **Tokenization**: The process of breaking text into individual units (tokens), typically words or subwords. In this project, we use NLTK's word tokenizer to split news articles into individual words.

- **Stopwords**: Common words (like "the", "a", "an") that carry little meaningful information. We remove these using NLTK's stopwords list to reduce noise in our data.

- **Padding and Sequences**: Neural networks require fixed-length inputs. After converting words to numbers (tokenization), we use padding to ensure each article representation has the same length by adding zeros where needed (pad_sequences).

- **Embeddings**: A technique to represent words as dense vectors in a continuous vector space. Words with similar meanings have similar vector representations, allowing the model to understand semantic relationships.

## Overview

This project implements a deep learning model using bidirectional LSTM networks to classify news articles as either real or fake. The model analyzes the text content and title of news articles to determine their authenticity.

## Dataset

The dataset consists of two CSV files:

- `True.csv`: Contains real news articles with their titles and text
- `Fake.csv`: Contains fake news articles with their titles and text

## Requirements

The project requires the following Python libraries:

- **numpy**: For numerical operations and array handling; provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these
- **pandas**: For data manipulation and analysis; offers data structures like DataFrames for efficiently storing and manipulating datasets
- **tensorflow**: For building and training deep learning models; an open-source machine learning framework that provides tools for neural network implementation
- **scikit-learn**: For model evaluation metrics and data splitting; includes tools for data preprocessing, model selection, and evaluation
- **matplotlib**: For data visualization; a comprehensive library for creating static, interactive, and animated visualizations
- **nltk** (Natural Language Toolkit): For natural language processing tasks; provides tools for text processing including tokenization, stemming, tagging, and parsing
- **keras**: For building neural networks (part of TensorFlow); a high-level neural networks API that simplifies the creation of deep learning models

These dependencies can be installed using:

```
pip install -r requirements.txt
```

## Code Explanation

### Cell 1: Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
import re
import string
```

This cell imports all the necessary libraries for data manipulation, neural network building, evaluation, and text processing.

### Cell 2: Downloading NLTK Resources

```python
nltk.download('stopwords')
nltk.download('punkt')
```

This cell downloads the NLTK stopwords and punkt tokenizer models which are used for text preprocessing.

### Cell 3: Downloading Additional NLTK Resource

```python
import nltk
nltk.download('punkt_tab')
```

This cell downloads the punkt_tab resource which is required for NLTK's word tokenization functionality.

### Cell 4: Loading and Labeling Data

```python
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1  # 1 for true news
fake_news['label'] = 0  # 0 for fake news

print(f"True news shape: {true_news.shape}")
print(f"Fake news shape: {fake_news.shape}")
```

This cell loads the true and fake news datasets from CSV files and adds label columns (1 for true news, 0 for fake news).

### Cell 5 & 6: Data Exploration

```python
true_news.head()
fake_news.head()
```

These cells display the first few rows of each dataset to understand their structure.

### Cell 7: Combining and Shuffling Data

```python
news_dataset = pd.concat([true_news, fake_news], ignore_index=True)
news_dataset = news_dataset.sample(frac=1).reset_index(drop=True)
print(f"Total dataset shape: {news_dataset.shape}")
```

This cell combines the true and fake news datasets and randomly shuffles the combined dataset to ensure good training distribution.

### Cell 8: Text Cleaning Function

```python
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text)

    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    return text
```

This cell defines a function to clean and preprocess text data by:

1. Converting text to lowercase
2. Removing text within square brackets
3. Removing punctuation
4. Removing words containing numbers
5. Removing extra whitespace
6. Tokenizing the text
7. Removing stopwords
8. Rejoining the words

### Cell 9: Data Preprocessing

```python
news_dataset['cleaned_text'] = news_dataset['text'].apply(clean_text)
news_dataset['cleaned_title'] = news_dataset['title'].apply(clean_text)
news_dataset['combined_content'] = news_dataset['cleaned_title'] + ' ' + news_dataset['cleaned_text']

X = news_dataset['combined_content']
y = news_dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
```

This cell:

1. Applies text cleaning to both the article text and titles
2. Combines the cleaned title and text
3. Splits the dataset into features (X) and labels (y)
4. Further splits the data into training (80%) and testing (20%) sets

### Cell 10: Text Tokenization and Sequencing

```python
max_words = 50000
max_sequence_length = 300

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens')

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
```

This cell:

1. Converts the text data into numerical sequences using tokenization
2. Limits vocabulary to 50,000 most frequent words
3. Adds an out-of-vocabulary token for unknown words
4. Converts the sequences to equal-length padded sequences (300 tokens)

### Cell 11: Model Architecture

```python
embedding_dim = 128
vocab_size = min(max_words, len(word_index) + 1)

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.summary()
```

This cell defines the neural network architecture:

1. An embedding layer to learn word representations (128 dimensions)
2. A bidirectional LSTM layer with 64 units that returns sequences
3. A dropout layer (30%) to prevent overfitting
4. Another bidirectional LSTM layer with 32 units
5. Another dropout layer (30%)
6. A dense hidden layer with 16 units and ReLU activation
7. A final dropout layer (30%)
8. A single output neuron with sigmoid activation for binary classification

### Cell 12: Model Compilation

```python
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
```

This cell configures the model training process with:

1. Binary cross-entropy loss (suitable for binary classification)
2. Adam optimizer with a learning rate of 0.001
3. Accuracy as the evaluation metric

### Cell 13: Model Training

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train_padded, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping]
)
```

This cell:

1. Sets up early stopping to prevent overfitting by monitoring validation loss
2. Trains the model for up to 20 epochs with a batch size of 64
3. Uses 10% of the training data for validation
4. Automatically stops training if validation loss doesn't improve for 3 consecutive epochs

### Cell 14: Model Evaluation

```python
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
```

This cell evaluates the model on the test dataset and reports the final loss and accuracy.

### Cell 15: Detailed Evaluation

```python
y_pred_proba = model.predict(X_test_padded)
y_pred = (y_pred_proba > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

This cell:

1. Gets model predictions for the test dataset
2. Converts probability outputs to binary predictions
3. Generates a detailed classification report including precision, recall, and F1-score
4. Creates a confusion matrix showing true positives, false positives, true negatives, and false negatives

### Cell 16: Confusion Matrix Visualization

```python
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Fake News', 'True News']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
```

This cell creates a visual representation of the confusion matrix to easily interpret model performance.

### Cell 17: Model and Tokenizer Saving

```python
model.save('fake_news_detector.h5')
print("Model saved as 'fake_news_detector.h5'")

import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved as 'tokenizer.pickle'")
```

This cell saves:

1. The trained model to a HDF5 file for future use
2. The fitted tokenizer to a pickle file to ensure consistent text preprocessing

### Cell 18: Example Predictions

```python
def predict_news(news_text, model, tokenizer, max_seq_length=300):
    cleaned_text = clean_text(news_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)[0][0]
    result = "True News" if prediction > 0.5 else "Fake News"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return result, confidence * 100

sample_fake_news = fake_news.iloc[0]['text']
sample_true_news = true_news.iloc[0]['text']

result_fake, confidence_fake = predict_news(sample_fake_news, model, tokenizer)
result_true, confidence_true = predict_news(sample_true_news, model, tokenizer)

print(f"Sample Fake News Prediction: {result_fake} with {confidence_fake:.2f}% confidence")
print(f"Sample True News Prediction: {result_true} with {confidence_true:.2f}% confidence")
```

This cell:

1. Defines a function to predict whether a given text is fake or true news
2. Tests the function on one sample from each dataset
3. Displays the prediction result and confidence level for each sample

## How to Use the Model

After running all cells, you can use the `predict_news` function with any news text to classify it as either fake or true. The function returns both the classification and a confidence percentage.

## Model Performance

The model's performance can be assessed through:

1. Test accuracy
2. Classification report metrics (precision, recall, F1-score)
3. The confusion matrix visualization

Typical good models should achieve above 90% accuracy on the test set.
