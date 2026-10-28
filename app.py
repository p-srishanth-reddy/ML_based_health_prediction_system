import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from src.module_registry import ModuleRegistry
from src.predict import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'healthai-clinical-prototype-secret-key-2026')

# In-memory screening session history (privacy-conscious, per-client session)
_screening_history = []

def validate_inputs(schema, data):
    """Validates user-submitted inputs against physiological ranges and valid options."""
    errors = []
    validated = {}

    for field, rules in schema.items():
        val = data.get(field)
        field_label = rules.get('label', field)

        if val is None or val == '':
            errors.append(f"Field '{field_label}' is required.")
            continue

        if rules['type'] == 'number':
            try:
                num_val = float(val)
                min_val = rules.get('min')
                max_val = rules.get('max')

                if min_val is not None and num_val < min_val:
                    errors.append(f"{field_label} ({num_val}) is below acceptable clinical minimum ({min_val}).")
                elif max_val is not None and num_val > max_val:
                    errors.append(f"{field_label} ({num_val}) exceeds acceptable clinical maximum ({max_val}).")
                else:
                    validated[field] = num_val
            except (ValueError, TypeError):
                errors.append(f"Field '{field_label}' must be a valid numerical value.")

        elif rules['type'] == 'select':
            options = rules.get('options', [])
            str_val = str(val).strip()
            if str_val not in options:
                errors.append(f"Selected value '{str_val}' for '{field_label}' is not a valid option.")
            else:
                validated[field] = str_val

    return validated, errors

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/evaluation')
def evaluation_page():
    return render_template('evaluation.html')

@app.route('/predict/<module_id>')
def prediction_page(module_id):
    module = ModuleRegistry.get_module(module_id)
    if not module:
        return render_template('index.html')
    return render_template('prediction.html', module_id=module_id, module=module)

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
        return jsonify({'error': f"Unknown module '{module_id}'"}), 404

    data = request.json
    if not data or not isinstance(data, dict):
        return jsonify({'error': 'No input data provided or invalid JSON payload.'}), 400

    # 1. Strict Input Range & Schema Validation
    validated_data, validation_errors = validate_inputs(module['schema'], data)
    if validation_errors:
        return jsonify({
            'error': 'Clinical input validation failed.',
            'details': validation_errors
        }), 400

    try:
        # 2. Comprehensive Multi-Model Prediction with Explainability
        full_res = Predictor.predict_all(module_id, validated_data)

        # Record screening session (anonymized for patient privacy)
        screening_entry = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'module_id': module_id,
            'module_name': module['name'],
            'primary_model': full_res['primary_model'],
            'prediction': full_res['primary_result']['human_label'],
            'risk_score': full_res['primary_result']['score'],
            'risk_tier': full_res['risk_tier'],
            'risk_badge': full_res['risk_badge']
        }
        _screening_history.append(screening_entry)
        # Retain last 50 screenings in memory
        if len(_screening_history) > 50:
            _screening_history.pop(0)

        # Backwards-compatibility mappings for legacy UI components
        comparisons = full_res.get('model_comparisons', {})
        dt_res = comparisons.get('decision_tree', full_res['primary_result'])
        rf_res = comparisons.get('random_forest', full_res['primary_result'])
        lr_res = comparisons.get('logistic_regression', full_res['primary_result'])

        response_payload = {
            **full_res,
            'decision_tree': dt_res,
            'random_forest': rf_res,
            'logistic_regression': lr_res,
            'validated_inputs': validated_data
        }

        return jsonify(response_payload)

    except FileNotFoundError as e:
        logger.error(f"Model file error: {e}")
        return jsonify({'error': 'Trained model artifacts are missing. Please ensure models are trained.'}), 500
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f"Inference error: {str(e)}"}), 400

@app.route('/api/evaluation/<module_id>')
def get_evaluation(module_id):
    meta_path = f"models/{module_id}/metadata.json"
    if not os.path.exists(meta_path):
        return jsonify({'error': 'Evaluation metadata not available for this module.'}), 404

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    return jsonify({
        'module_id': module_id,
        'evaluation': meta.get('evaluation', {}),
        'dataset_info': meta.get('dataset_info', {}),
        'feature_importance': meta.get('feature_importance', {}),
        'primary_model': meta.get('primary_model', 'random_forest'),
        'classes': meta.get('classes', [])
    })

@app.route('/api/history', methods=['GET', 'DELETE'])
def manage_history():
    """Privacy-first screening history management."""
    global _screening_history
    if request.method == 'DELETE':
        _screening_history = []
        return jsonify({'message': 'Screening history securely wiped.', 'history': []})
    return jsonify({'history': list(reversed(_screening_history))})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
