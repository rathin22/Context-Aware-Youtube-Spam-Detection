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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode
    embeddings = []

    # Assuming 'comments' is a list of text comments
    for comment in comments:
        # Tokenize the comment and move tokens to GPU
        tokens = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Perform the forward pass and move the output tensor to CPU
        with torch.no_grad():
            outputs = model(**tokens)
            # Extract the embeddings of the [CLS] token, and move back to CPU
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding)

    # Convert the list of embeddings into a numpy array
    return np.vstack(embeddings)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)


EXTERNAL_DATA = 1

if not EXTERNAL_DATA:
    data = pd.read_csv('saved_data/eminem_comments.csv')
    print("\nTotal no. of comments:", len(data))

    # Remove non-english comments
    data = data[data['non_english'] == 0]
    print("No. of english comments:", len(data))
    print("\nNo. of spam comments:", len(data[data['spam_with_context'] == 1]))
    print("No. of non-spam comments:", len(data[data['spam_with_context'] == 0]))

    # Reducing the number of samples corresponding to a majority class
    non_spam_sample = data[data['spam_with_context'] == 0].sample(n=600, random_state=23)
    data = data.drop(non_spam_sample.index)

    print("\nAfter undersampling:")
    print("No. of spam comments:", len(data[data['spam_with_context'] == 1]))
    print("No. of non-spam comments:", len(data[data['spam_with_context'] == 0]))

else:
    # Use external dataset instead
    data = pd.read_csv('saved_data/Youtube04-Eminem.csv')
    data = data.rename(columns={'CONTENT': 'comment_text', 'CLASS': 'spam_with_context'})
    print("\nTotal no. of comments:", len(data))
    print("\nNo. of spam comments:", len(data[data['spam_with_context'] == 1]))
    print("No. of non-spam comments:", len(data[data['spam_with_context'] == 0]))
    
data['comment_text'] = data['comment_text'].fillna("")

# Splitting the dataset into training and testing sets
X = data['comment_text']
y = data['spam_with_context']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))

# Vectorize the comment text
X_train_vec = vectorize_comments(X_train.tolist())
X_test_vec = vectorize_comments(X_test.tolist())

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
