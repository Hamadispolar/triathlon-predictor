"""
Triathlon Predictor — Flask API
================================
Serves predictions from the trained ML model via a REST endpoint.

Usage:
    python app.py

Endpoints:
    POST /predict   — predict finish time
    GET  /metrics   — model performance metrics
    GET  /health    — health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import traceback

from model import predict, train, generate_dataset

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model.pkl'

# Auto-train on first run if no model exists
if not os.path.exists(MODEL_PATH):
    print("No model found — training on synthetic data...")
    train(generate_dataset(n=600), save_path=MODEL_PATH)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': os.path.exists(MODEL_PATH)})


@app.route('/metrics', methods=['GET'])
def metrics():
    if os.path.exists('metrics.json'):
        with open('metrics.json') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'metrics not found'}), 404


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        data = request.get_json()
        required = [
            'weekly_swim_km', 'weekly_bike_km', 'weekly_run_km',
            'avg_swim_pace_min_per_100m', 'avg_bike_speed_kmh',
            'avg_run_pace_min_per_km', 'weeks_of_training',
            'races_completed', 'age', 'resting_hr', 'level'
        ]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        result = predict(data, model_path=MODEL_PATH)
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model (useful if you add real data later)."""
    try:
        df = generate_dataset(n=600)
        _, metrics_out = train(df, save_path=MODEL_PATH)
        return jsonify({'success': True, 'metrics': metrics_out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
