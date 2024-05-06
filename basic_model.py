from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from common_functions import load_data

EXTERNAL_DATA = 1

data = load_data(0, EXTERNAL_DATA)

# Splitting the dataset into training and testing sets
X = data['comment_text']
y = data['spam_with_context']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))

# Creating a model pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes Classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Training the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
print("\nAccuracy:", accuracy)
print()
print(report)
