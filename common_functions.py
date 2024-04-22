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

def get_mean_embedding_excluding_padding(model, tokenizer, text):
    device = "cuda"
    # Tokenize with padding=True to handle variable-length texts
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    # Move all tensors in encoded_input to device
    encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}  
    attention_mask = encoded_input['attention_mask']
    
    # Generate embeddings
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Use attention_mask to exclude PAD tokens
    mask_expanded = attention_mask.unsqueeze(-1).expand(output.last_hidden_state.size()).to(device)
    sum_embeddings = torch.sum(output.last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings.squeeze().cpu()

def vectorize_comments(comments, model, tokenizer, video_titles=None):
    def get_embedding(comment):
        device = "cuda"
        tokens = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**tokens)
            # Extract the embeddings of the [CLS] token, and move back to CPU
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embedding

    model.eval()  # Set the model to evaluation mode
    embeddings = []
    token_lengths = []
    saved_context_embeddings = {}
    # Assuming 'comments' is a list of text comments
    if video_titles is not None:
        for comment, title in zip(comments, video_titles):
            if title in saved_context_embeddings:
                embeddings.append(saved_context_embeddings[title])
                #print("Using saved embedding")
            else:
                embedding = get_mean_embedding_excluding_padding(model, tokenizer, comment)
                embeddings.append(embedding)
                saved_context_embeddings[title] = embedding
    else:
        embeddings = [get_mean_embedding_excluding_padding(model, tokenizer, comment) for comment in comments]
    
    # Convert the list of embeddings into a numpy array
    return np.vstack(embeddings)

def sf_encode(texts, model, video_titles=None):
    if video_titles is None:
        return model.encode(texts)

    saved_context_embeddings = {}
    embeddings = []
    for text, title in zip(texts, video_titles):
        if title in saved_context_embeddings:
            embeddings.append(saved_context_embeddings[title])
        else:
            embedding = model.encode(text)
            embeddings.append(embedding)
            saved_context_embeddings[title] = embedding
    
    return np.vstack(embeddings)
