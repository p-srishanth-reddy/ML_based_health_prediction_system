import pandas as pd
import numpy as np
import json
import os
from flask import Flask, render_template, request, jsonify
from src.module_registry import ModuleRegistry
from src.predict import Predictor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/modules')
def get_modules():
    return jsonify(ModuleRegistry.list_modules())

@app.route('/api/modules/<module_id>/schema')
def get_schema(module_id):
    module = ModuleRegistry.get_module(module_id)
    if not module:
        return jsonify({'error': 'Module not found'}), 404
    return jsonify(module['schema'])

@app.route('/api/predict/<module_id>', methods=['POST'])
def predict(module_id):
    module = ModuleRegistry.get_module(module_id)
    if not module:
        return jsonify({'error': 'Module not found'}), 404

    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # We can predict with both DT and RF and return both
        dt_res = Predictor.predict(module_id, data, model_type='decision_tree')
        rf_res = Predictor.predict(module_id, data, model_type='random_forest')

        return jsonify({
            'decision_tree': dt_res,
            'random_forest': rf_res,
            'primary_model': 'random_forest'
        })
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/evaluation/<module_id>')
def get_evaluation(module_id):
    model_dir = f"models/{module_id}"
    meta_path = f"{model_dir}/metadata.json"

    if not os.path.exists(meta_path):
        return jsonify({'error': 'Evaluation data not available. Please train models.'}), 404

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    return jsonify(meta['evaluation'])

@app.route('/evaluation')
def evaluation_page():
    return render_template('evaluation.html')

@app.route('/predict/<module_id>')
def prediction_page(module_id):
    return render_template('prediction.html', module_id=module_id)

if __name__ == '__main__':
    app.run(debug=True)
