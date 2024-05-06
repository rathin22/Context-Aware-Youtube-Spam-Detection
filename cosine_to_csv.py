from sklearn.model_selection import train_test_split
from common_functions import load_data, vectorize_comments, sf_encode
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda"
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model = SentenceTransformer('all-MiniLM-L12-v2')

data = load_data(1, 1)

data['combined_context'] = (
                            data['video_title'] + " " +
                            data['video_description'] + " " +
                            data['video_tags'] + " " +
                            data["video_category"]
                            )

X = data[['comment_text', 'combined_context', 'video_title']]
y = data['spam_with_context']
X.reset_index(drop=True)

# Vectorize the comment text using BERT
#X_comment_vec = vectorize_comments(X['comment_text'], model, tokenizer)
#X_context_vec = vectorize_comments(X['combined_context'], model, tokenizer,  X['video_title'])
X_comment_vec = sf_encode(X['comment_text'], model)
X_context_vec = sf_encode(X['combined_context'], model, X['video_title'])

# Calculate the cosine similarity between comment text and combined context
#X_context_sim = cosine_similarity(X_comment_vec, X_context_vec)
X_context_sim = util.cos_sim(X_comment_vec, X_context_vec)
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