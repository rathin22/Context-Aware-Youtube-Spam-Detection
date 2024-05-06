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
from common_functions import load_data, vectorize_comments
from joblib import dump, load
import time

def main():
    device = "cuda"
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased').to(device)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model_without_context(model, tokenizer)

def model_without_context(model, tokenizer):
    start_time = time.time()
    data = load_data(include_context=0, external_data=1)
        
    # Splitting the dataset into training and testing sets
    X = data['comment_text']
    y = data['spam_with_context']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the comment text
    X_train_vec = vectorize_comments(X_train.tolist(), model, tokenizer)
    X_test_vec = vectorize_comments(X_test.tolist(), model, tokenizer)

    # Apply PCA to reduce the dimensionality of the comment embeddings
    # pca = PCA(n_components=50, random_state=42)  # Adjust the number of components as needed
    # X_train_vec_pca = pca.fit_transform(X_train_vec)
    # X_test_vec_pca = pca.transform(X_test_vec)

    # Initialize the classifier
    classifier = LogisticRegression(max_iter=1000)

    # Training the model
    classifier.fit(X_train_vec, y_train)

    # Save the classifier
    #dump(classifier, 'classifier.joblib')

    # Predicting on the test set
    y_pred = classifier.predict(X_test_vec)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print(report)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken: {duration} seconds")
