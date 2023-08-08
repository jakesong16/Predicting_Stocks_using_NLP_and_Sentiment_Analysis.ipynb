from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from keras.preprocessing.sequence import pad_sequences
import torch
import numpy as np

# Load model architecture
model_config_path = 'bert_model_config.json'
model_config = BertConfig.from_json_file(model_config_path)
model = BertForSequenceClassification(model_config)

# Load model weights
model_weights_path = 'bert_model_weights.pt'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

app = Flask(__name__)

# Define your label map
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_text = data['text']

        # Preprocess input and make predictions
        tokenized_text = tokenizer.tokenize("[CLS] " + input_text + " [SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
        input_ids = pad_sequences([input_ids], maxlen=128, dtype="long", truncating="post", padding="post")
        attention_mask = [1.0 if value > 0 else 0.0 for value in input_ids[0]]

        input_tensor = torch.tensor(input_ids)
        attention_mask_tensor = torch.tensor([attention_mask])

        with torch.no_grad():
            outputs = model(input_tensor, attention_mask=attention_mask_tensor)
            logits = outputs.logits

        predicted_label = torch.argmax(logits[0]).item()
        sentiment = LABEL_MAP[predicted_label]
        
        return jsonify({'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
       
