from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

def vectorize_comments(comments):
    # Tokenize the comments. This will also truncate or pad the sequences to a fixed length.
    tokens = tokenizer(comments.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Generate embeddings for each token. For simplicity, we'll use the embeddings of the [CLS] token
    # as the representation of the entire comment.
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state[:,0,:].numpy()  # Extract the embeddings of the [CLS] token
    
    return embeddings

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

data = pd.read_csv('saved_data\eminem_comments.csv')
print("Total no. of comments:", len(data))

external_data = pd.read_csv('saved_data/eminem_comments.csv')
external_data = external_data.rename(columns={'CONTENT': 'comment_text', 'CLASS': 'spam_with_context'})

# Remove non-english comments
data = data[data['non_english'] == 0]
print("No. of english comments:", len(data))

print("\nNo. of spam comments:", len(data[data['spam_with_context'] == 1]))
print("No. of non-spam comments:", len(data[data['spam_with_context'] == 0]))

# Reducing the number of samples corresponding to a majority class
non_spam_sample = data[data['spam_with_context'] == 0].sample(n=600)
data = data.drop(non_spam_sample.index)

print("\nAfter undersampling:")
print("No. of spam comments:", len(data[data['spam_with_context'] == 1]))
print("No. of non-spam comments:", len(data[data['spam_with_context'] == 0]))

data['comment_text'] = data['comment_text'].fillna("")

# Splitting the dataset into training and testing sets
X = data['comment_text']
y = data['spam_with_context']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))

# Vectorize the comment text
X_train_vec = vectorize_comments(X_train)
X_test_vec = vectorize_comments(X_test)

# Initialize the classifier
classifier = LogisticRegression(max_iter=1000)

# Training the model
classifier.fit(X_train_vec, y_train)

# Predicting on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print()
print(report)
