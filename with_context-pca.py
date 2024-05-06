# COSINE SIMILARITY HAS BEEN ADDED TO transfomers_with_context.py AS WELL
# ONLY EXTRA THING IN THIS FILE IS THE PCA REDUCTION

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from common_functions import load_data, vectorize_comments
from sentencetransformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
print(f"Using device: {device}")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
#model = SentenceTransformer('bert-base-nli-mean-tokens')

data = load_data(1, 1)

data['combined_context'] = (
                            data['video_title'] + " " +
                            data['video_description'] + " " +
                            data['video_tags'] + " " +
                            data["video_category"]
                            )

X = data[['comment_text', 'combined_context', 'video_title']]
y = data['spam_with_context']
print("X")
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))
print("X_train")
print(X_train.shape)

# Vectorize the comment text using BERT
X_train_comment_vec = vectorize_comments(X_train['comment_text'], model, tokenizer)
X_test_comment_vec = vectorize_comments(X_test['comment_text'], model, tokenizer)
print("X_train_comment_vec")
print(X_train_comment_vec.shape)

# Vectorize the combined contextual information using BERT
X_train_context_vec = vectorize_comments(X_train['combined_context'], model, tokenizer, X_train['video_title'])
X_test_context_vec = vectorize_comments(X_test['combined_context'], model, tokenizer, X_test['video_title'])
print("X_train_context_vec")
print(X_train_context_vec.shape)

# Apply PCA to reduce the dimensionality of the comment embeddings
pca = PCA(n_components=50, random_state=42)  # Adjust the number of components as needed
X_train_comment_vec_pca = pca.fit_transform(X_train_comment_vec)
X_test_comment_vec_pca = pca.transform(X_test_comment_vec)


# Create the feature matrices by stacking the BERT embeddings and cosine similarity scores
X_train_vec = np.hstack((X_train_comment_vec_pca, X_train_context_vec))
X_test_vec = np.hstack((X_test_comment_vec_pca, X_test_context_vec))
print("X_train_vec")
print(X_train_vec.shape)

# Initialize the classifier
classifier = LogisticRegression(max_iter=1000)

# Training the model
classifier.fit(X_train_vec, y_train)

# Predicting on the test set
y_pred = classifier.predict(X_test_vec)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print("\nAccuracy:", accuracy)
print()
print(report)