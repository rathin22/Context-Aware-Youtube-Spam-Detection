from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
from transformers import BertTokenizer, BertModel
import sys
import os
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
common_functions_dir = os.path.join(current_dir_path, "..", "..")
sys.path.append(common_functions_dir)
from common_functions import vectorize_comments

app = Flask(__name__)
CORS(app)
#CORS(app, origins=["https://www.youtube.com"])
#CORS(app, resources={r"/predict": {"origins": ["https://www.youtube.com"]}}, allow_headers=["Content-Type"])

device = "cuda"

# Load your trained model
classifier = joblib.load('yt-extension/server/classifier.joblib')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the comment from the request's JSON body
    data = request.get_json(force=True)
    comments = data['comments']
    print("Comments:")
    print(comments)
    print()

    comments_vectors = vectorize_comments(list(comments), bert_model, bert_tokenizer)
    
    # Predict
    predictions = classifier.predict(comments_vectors)
    print("Predictions:", predictions)
    # Return a JSON response indicating spam or not spam
    response = jsonify({'spam': [bool(pred) for pred in predictions]})
    print("Response:", response)
    # response.headers.add('Access-Control-Allow-Origin', 'https://www.youtube.com')
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
