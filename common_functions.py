import pandas as pd
import torch
import numpy as np

def load_data(include_context=0, external_data=0):

    if not external_data:
        data = pd.read_csv('saved_data/eminem_comments.csv')
        data['comment_text'] = data['comment_text'].fillna("")

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

        context_file_path = 'saved_data/eminem_video_context.csv'

    else:
        # Use external dataset instead
        data = pd.read_csv('saved_data/online_dataset/combined_data.csv')
        data = data.drop(['COMMENT_ID', 'AUTHOR'], axis=1)
        data = data.rename(columns={'CONTENT': 'comment_text', 'CLASS': 'spam_with_context'})
        print("\nTotal no. of comments:", len(data))
        print("\nNo. of spam comments:", len(data[data['spam_with_context'] == 1]))
        print("No. of non-spam comments:", len(data[data['spam_with_context'] == 0]))

        context_file_path = 'saved_data/online_dataset/video_contexts.csv'

    if include_context:
        context_data = pd.read_csv(context_file_path)
        context_data = context_data.drop(['video_thumbnail'], axis=1)
        # Merge the comment data with the video details based on video_id
        data = data.merge(context_data, on='video_id', how='left')
        print("merged data")
        
    print(data.shape)
    return data

def vectorize_comments(comments, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode
    embeddings = []
    token_lengths = []
    # Assuming 'comments' is a list of text comments
    for comment in comments:
        # full_tokens = tokenizer(comment)
        # token_length = len(full_tokens['input_ids'])
        # token_lengths.append(token_length)
        # if token_length > 512:
        #    print("TOO MUCH")
        #    print(comment)
        #    print()
        # Tokenize the comment and move tokens to GPU
        tokens = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Perform the forward pass and move the output tensor to CPU
        with torch.no_grad():
            outputs = model(**tokens)
            # Extract the embeddings of the [CLS] token, and move back to CPU
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(embedding)
    #print("Minimum and max token length:")
    #print(min(token_lengths))
    #print(max(token_lengths))
    
    # Convert the list of embeddings into a numpy array
    return np.vstack(embeddings)
