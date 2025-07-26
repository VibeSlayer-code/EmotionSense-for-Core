import os
import torch
from flask import Flask, request, jsonify
from train_model import EmotionClassifier
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})


model_path = os.getenv(r"PUT emotion_predictor_model.py PATH HERE")
try:
    classifier = EmotionClassifier(
        model_name='roberta-large',
        checkpoint_path=model_path
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = classifier.predict_emotion(text, use_ensemble=True)
        
        response = {
            'predicted_emotion': result['emotion'],
            'confidence': result['confidence']
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
