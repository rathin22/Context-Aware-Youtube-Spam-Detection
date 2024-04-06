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
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from common_functions import load_data, vectorize_comments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"
print(f"Using device: {device}")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

data = load_data(1)

X = data[['comment_text', 'video_title', 'video_description', 'video_tags', 'video_category']]
print("X")
print(X.shape)
y = data['spam_with_context']

# Vectorize the comment text using BERT
X_comment_vec = vectorize_comments(X['comment_text'].tolist())

# Concatenate the contextual information
X_context = X['video_title'] + ' ' + X['video_tags'] + ' ' + X['video_category'] + ' ' + X['video_description']
# Vectorize the combined contextual information using BERT
X_context_vec = vectorize_comments(X_context.tolist())

# Calculate the cosine similarity between comment text and combined context
X_context_sim = cosine_similarity(X_comment_vec, X_context_vec)

# Take the diagonal elements of the similarity matrix
X_context_sim = np.diagonal(X_context_sim)

# Create a DataFrame with comments, cosine similarity scores, and labels
df_scores = pd.DataFrame({
    'comment_text': X['comment_text'],
    'cosine_similarity': X_context_sim,
    'label': y
})

# Map the label values to 'spam' and 'non-spam'
df_scores['label'] = df_scores['label'].map({1: 'spam', 0: 'non-spam'})

# Save the DataFrame to a CSV file
output_file = 'cosine_scores.csv'
df_scores.to_csv(output_file, index=False)

print(f"Cosine similarity scores and comments saved to '{output_file}'.")
exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))
print("X_train")
print(X_train.shape)
# Vectorize the comment text using BERT
X_train_comment_vec = vectorize_comments(X_train['comment_text'].tolist())
X_test_comment_vec = vectorize_comments(X_test['comment_text'].tolist())
print("X_train_comment_vec")
print(X_train_comment_vec.shape)

# Apply PCA to reduce the dimensionality of the comment embeddings
pca = PCA(n_components=50, random_state=42)  # Adjust the number of components as needed
X_train_comment_vec_pca = pca.fit_transform(X_train_comment_vec)
X_test_comment_vec_pca = pca.transform(X_test_comment_vec)

# Concatenate the contextual information
X_train_context = X_train['video_title'] + ' ' + X_train['video_tags'] + ' ' + X_train['video_category'] + ' ' + X_train['video_description']
X_test_context = X_test['video_title'] + ' ' + X_test['video_tags'] + ' ' + X_test['video_category'] + ' ' + X_test['video_description']
print("X_train_context")
print(X_train_context.shape)

# Vectorize the combined contextual information using BERT
X_train_context_vec = vectorize_comments(X_train_context.tolist())
X_test_context_vec = vectorize_comments(X_test_context.tolist())
print("X_train_context_vec")
print(X_train_context_vec.shape)

# Calculate the cosine similarity between comment text and combined context
X_train_context_sim = cosine_similarity(X_train_comment_vec, X_train_context_vec)
X_test_context_sim = cosine_similarity(X_test_comment_vec, X_test_context_vec)
print("X_train_context_sim")
print(X_train_context_sim.shape)

# Take the diagonal elements of the similarity matrices
X_train_context_sim = np.diagonal(X_train_context_sim).reshape(-1, 1)
X_test_context_sim = np.diagonal(X_test_context_sim).reshape(-1, 1)
print(X_train_context_sim)

# Create the feature matrices by stacking the BERT embeddings and cosine similarity scores
X_train_vec = np.hstack((X_train_comment_vec, X_train_context_sim))
X_test_vec = np.hstack((X_test_comment_vec, X_test_context_sim))
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
report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print()
print(report)