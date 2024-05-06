from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import sys
import os
import numpy as np
import json
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
common_functions_dir = os.path.join(current_dir_path, "..", "..")
sys.path.append(common_functions_dir)
from common_functions import vectorize_comments

# Get video category mapping
video_categories_file = '.../../saved_data/video_categories.json'
with open(video_categories_file, 'r') as json_file:
    video_categories = json.load(json_file)

app = Flask(__name__)
CORS(app)
#CORS(app, origins=["https://www.youtube.com"])
#CORS(app, resources={r"/predict": {"origins": ["https://www.youtube.com"]}}, allow_headers=["Content-Type"])

device = "cuda"

# Load your trained model
classifier = joblib.load('yt-extension/server/classifier.joblib')
classifier_with_context = joblib.load('yt-extension/server/classifier_with_context.joblib')
bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# Store video details in memory for simplicity; consider a database for production
context_store = {}

@app.route('/send_video_details', methods=['POST'])
def submit_video_details():
    data = request.get_json()
    video_id = data['videoId']
    print("Video details received")
    all_context = (
                    data['title'] + " " +
                    data['description'] + " " +
                    ' '.join(data['tags']) + " " +
                    video_categories[data['category']] 
                )
    print(all_context)
    if video_id not in context_store:
        context_vector = vectorize_comments([all_context], bert_model, bert_tokenizer)
        print(context_vector.shape)
        context_store[video_id] = context_vector
    return jsonify({'message': 'Video details saved', 'videoId': video_id})

@app.route('/predict', methods=['POST'])
def predict():
    # Get the comments from the request's JSON body
    data = request.get_json(force=True)
    video_id = data['videoId']
    comments = data['comments']
    video_details = context_store.get(video_id)
    if video_details is None:
        return jsonify({'error': 'Video details not found'}), 404
    
    print(comments)
    comments_vectors = vectorize_comments(list(comments), bert_model, bert_tokenizer)
    context_vector = context_store[video_id]
    print("context_vector shape")
    print(context_vector.shape)
    print(len(comments))
    context_vectors = np.repeat(context_vector, len(comments), axis=0)
    print("context_vectors shape")
    print(len(context_vectors))
    combined_vectors = np.concatenate((comments_vectors, context_vectors), axis=1)
    
    predictions = classifier_with_context.predict(combined_vectors)
    print("Predictions:", predictions)
    # Return a JSON response indicating spam or not spam
    response = jsonify({'spam': [bool(pred) for pred in predictions]})
    print("Response:", response)
    # response.headers.add('Access-Control-Allow-Origin', 'https://www.youtube.com')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
