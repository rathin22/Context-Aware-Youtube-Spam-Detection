from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from transformers import AlbertTokenizer, AlbertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from common_functions import load_data, vectorize_comments
import time

# Start timing
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
print(f"Using device: {device}")

# Initialize the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

data = load_data(include_context=1, external_data=1)

# data['combined_features'] = (data['video_title'] + " " +
#                             data['video_description'] + " " +
#                             data['video_tags'] + " " +
#                             data["video_category"] + " " +
#                             data['comment_text'] + " " +
#                             data['comment_text'] + " " +
#                             data['comment_text'])

data['combined_context'] = (
                            data['video_title'] + " " +
                            data['video_description'] + " " +
                            data['video_tags'] + " " +
                            data["video_category"]
                            )

# Splitting the dataset into training and testing sets
#X = data['combined_features']
X = data[['comment_text', 'combined_context']]
y = data['spam_with_context']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))
print("X_train")
print(X_train.shape)

# Vectorize the comment text and contextual information using BERT

# X_train_vec = vectorize_comments(X_train, model, tokenizer)
# X_test_vec = vectorize_comments(X_test, model, tokenizer)

X_train_comment_vec = vectorize_comments(X_train['comment_text'], model, tokenizer)
X_train_context_vec = vectorize_comments(X_train['combined_context'], model, tokenizer)
X_train_vec = np.concatenate((X_train_comment_vec, X_train_context_vec), axis=1)

X_test_comment_vec = vectorize_comments(X_test['comment_text'], model, tokenizer)
X_test_context_vec = vectorize_comments(X_test['combined_context'], model, tokenizer)
X_test_vec = np.concatenate((X_test_comment_vec, X_test_context_vec), axis=1)

# Initialize the classifier
classifier = LogisticRegression(max_iter=1000)

# Training the model
classifier.fit(X_train_vec, y_train)

# Predicting on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print()
print(report)

end_time = time.time()
duration = end_time - start_time
print(f"Time taken: {duration} seconds")