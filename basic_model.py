from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

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

# Creating a model pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes Classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Training the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)
print()
print(report)
