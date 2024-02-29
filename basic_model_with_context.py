# Re-importing pandas as the code execution state was reset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the context information dataset
context_file_path = 'saved_data/eminem_video_context.csv'
context_data = pd.read_csv(context_file_path)

EXTERNAL_DATA = 0

# Load the dataset
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
    comments_data = data.drop(non_spam_sample.index)

    print("\nAfter undersampling:")
    print("No. of spam comments:", len(comments_data[comments_data['spam_with_context'] == 1]))
    print("No. of non-spam comments:", len(comments_data[comments_data['spam_with_context'] == 0]))

else:
    # Use external dataset instead
    data = pd.read_csv('saved_data/Youtube04-Eminem.csv')

    # Eminem video id
    data['video_id'] = 'uelHwf8o7_U'

    comments_data = data.rename(columns={'CONTENT': 'comment_text', 'CLASS': 'spam_with_context'})
    print("\nTotal no. of comments:", len(comments_data))
    print("\nNo. of spam comments:", len(comments_data[comments_data['spam_with_context'] == 1]))
    print("No. of non-spam comments:", len(comments_data[comments_data['spam_with_context'] == 0]))

comments_data['comment_text'] = comments_data['comment_text'].fillna("")

# Merge the comments dataset with the context information based on the video ID
merged_data = comments_data.merge(context_data, left_on='video_id', right_on='video_id', how='left')

# Preparing the combined features: concatenating comment text with title, description, and tags

# Adding comment text twice to increase its weight
merged_data['combined_features'] = (merged_data['comment_text'] + " " +
                                    merged_data['comment_text'] + " " +
                                    merged_data['comment_text'] + " " +
                                    merged_data['comment_text'] + " " +
                                    merged_data['comment_text'] + " " +
                                    merged_data['comment_text'] + " " +
                                    merged_data['comment_text'] + " " +
                                    merged_data['comment_text'] + " " +
                                    merged_data['video_title'] + " " +
                                    merged_data['video_description'] + " " +
                                    merged_data['video_tags'] + " " +
                                    merged_data["video_category"])

# Splitting the dataset into training and testing sets
X = merged_data['combined_features']
y = merged_data['spam_with_context']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Number of spam comments in training set:", sum(y_train == 1))
print("Number of spam comments in test set:", sum(y_test == 1))

# Creating a model pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes Classifier
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


