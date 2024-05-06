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
from joblib import dump, load

def main():
    device = "cuda"
    # Initialize the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model_with_context(model, tokenizer)

def model_with_context(model, tokenizer):
    start_time = time.time()
    data = load_data(include_context=1, external_data=1)

    # data['combined_features'] = (
    #                             data['comment_text'] + " " +
    #                             data['comment_text'] + " " +
    #                             data['video_title'] + " " +
    #                             data['video_description'] + " " +
    #                             data['video_tags'] + " " +
    #                             data["video_category"]
    #                             )

    data['combined_context'] = (
                                data['video_title'] + " " +
                                data['video_tags'] + " " +
                                data["video_category"] + " " +
                                data['video_description']
                                )

    # Splitting the dataset into training and testing sets
    #X = data['combined_features']
    X = data[['comment_text', 'combined_context', 'video_title']]
    y = data['spam_with_context']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the comment text and contextual information using BERT
    X_train_comment_vec = vectorize_comments(X_train['comment_text'], model, tokenizer)
    X_train_context_vec = vectorize_comments(X_train['combined_context'], model, tokenizer, X_train['video_title'])

    X_test_comment_vec = vectorize_comments(X_test['comment_text'], model, tokenizer)
    X_test_context_vec = vectorize_comments(X_test['combined_context'], model, tokenizer, X_test['video_title'])

    # Calculate the cosine similarity between comment text and combined context
    X_train_cosine_sim = cosine_similarity(X_train_comment_vec, X_train_context_vec)
    X_test_cosine_sim = cosine_similarity(X_test_comment_vec, X_test_context_vec)
    # Take the diagonal elements of the similarity matrices
    X_train_cosine_sim = np.diagonal(X_train_cosine_sim).reshape(-1, 1)
    X_test_cosine_sim = np.diagonal(X_test_cosine_sim).reshape(-1, 1)

    # Calculate the length of comments in the training and test sets
    # X_train_length = np.array([len(comment) for comment in X_train['comment_text']]).reshape(-1, 1)
    # X_test_length = np.array([len(comment) for comment in X_test['comment_text']]).reshape(-1, 1)

    X_train_vec = np.concatenate((X_train_comment_vec, X_train_context_vec), axis=1)
    X_test_vec = np.concatenate((X_test_comment_vec, X_test_context_vec), axis=1)

    # X_train_vec = vectorize_comments(X_train, model, tokenizer)
    # X_test_vec = vectorize_comments(X_test, model, tokenizer)

    # Initialize the classifier
    classifier = LogisticRegression(max_iter=1000, C=1)

    # Training the model
    classifier.fit(X_train_vec, y_train)

    # Save the classifier
    #dump(classifier, 'classifier_with_context.joblib')

    # Predicting on the test set
    y_pred = classifier.predict(X_test_vec)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print(report)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken: {duration} seconds")

if __name__ == "__main__":
    main()