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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

def text_encode(texts, model, video_titles=None):
    if video_titles is None:
        return model.encode(texts)

    saved_context_embeddings = {}
    embeddings = []
    for text, title in zip(texts, video_titles):
        if title in saved_context_embeddings:
            embeddings.append(saved_context_embeddings[title])
        else:
            embedding = model.encode(text)
            embeddings.append(embedding)
            saved_context_embeddings[title] = embedding
    
    return np.vstack(embeddings)

# Start timing
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
print(f"Using device: {device}")

model = SentenceTransformer('all-MiniLM-L12-v2')

data = load_data(include_context=1, external_data=1)

data['combined_context'] = (
                            data['video_title'] + " " +
                            data['video_description'] + " " +
                            data['video_tags'] + " " +
                            data["video_category"]
                            )

# Splitting the dataset into training and testing sets
X = data[['comment_text', 'combined_context', 'video_title']]
y = data['spam_with_context']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))
print("X_train")
print(X_train.shape)

X_train = X_train.reset_index()
X_test = X_test.reset_index()

# Vectorize the comment text and contextual information using BERT
X_train_comment_vec = text_encode(X_train['comment_text'], model)
X_train_context_vec = text_encode(X_train['combined_context'], model, X_train['video_title'])
print("X_train_comment_vec")
print(X_train_comment_vec.shape)

X_test_comment_vec = text_encode(X_test['comment_text'], model)
X_test_context_vec = text_encode(X_test['combined_context'], model, X_test['video_title'])

# Calculate the cosine similarity between comment text and combined context
X_train_cosine_sim = util.cos_sim(X_train_comment_vec, X_train_context_vec)
X_test_cosine_sim = util.cos_sim(X_test_comment_vec, X_test_context_vec)
print("X_train_cosine_sim")
print(X_train_cosine_sim.shape)
# Take the diagonal elements of the similarity matrices
X_train_cosine_sim = np.diagonal(X_train_cosine_sim).reshape(-1, 1)
X_test_cosine_sim = np.diagonal(X_test_cosine_sim).reshape(-1, 1)
print("X_train_cosine_sim")
print(X_train_cosine_sim.shape)

X_train_vec = np.concatenate((X_train_comment_vec, X_train_context_vec), axis=1)
X_test_vec = np.concatenate((X_test_comment_vec, X_test_context_vec), axis=1)

# Initialize the classifier
classifier = LogisticRegression(max_iter=1000, C=1)

# Training the model
classifier.fit(X_train_comment_vec, y_train)

# Predicting on the test set
y_pred = classifier.predict(X_test_comment_vec)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print()
print(report)

end_time = time.time()
duration = end_time - start_time
print(f"Time taken: {duration} seconds")