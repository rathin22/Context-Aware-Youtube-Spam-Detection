# Re-importing pandas as the code execution state was reset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from common_functions import load_data


EXTERNAL_DATA = 1

comments_data = load_data(1, EXTERNAL_DATA)

comments_data['comment_text'] = comments_data['comment_text'].fillna("")

# Merge the comments dataset with the context information based on the video ID
#comments_data = comments_data.merge(context_data, left_on='video_id', right_on='video_id', how='left')

# Preparing the combined features: concatenating comment text with title, description, and tags

# Adding comment text twice to increase its weight
comments_data['combined_features'] = (comments_data['comment_text'] + " " +
                                    comments_data['comment_text'] + " " +
                                    comments_data['comment_text'] + " " +
                                    comments_data['comment_text'] + " " +
                                    comments_data['comment_text'] + " " +
                                    comments_data['comment_text'] + " " +
                                    comments_data['comment_text'] + " " +
                                    comments_data['comment_text'] + " " +
                                    comments_data['video_title'] + " " +
                                    comments_data['video_description'] + " " +
                                    comments_data['video_tags'] + " " +
                                    comments_data["video_category"])

# Splitting the dataset into training and testing sets
X = comments_data['combined_features']
y = comments_data['spam_with_context']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a model pipeline with Bag-of-Words and Multinomial Naive Bayes Classifier
model_with_context = make_pipeline(CountVectorizer(), MultinomialNB())

# Training the model
model_with_context.fit(X_train, y_train)

# Predicting on the test set
y_pred_with_context = model_with_context.predict(X_test)

# Evaluating the model
accuracy_with_context = accuracy_score(y_test, y_pred_with_context)
report_with_context = classification_report(y_test, y_pred_with_context)
print("\nAccuracy:", accuracy_with_context)
print()
print(report_with_context)


