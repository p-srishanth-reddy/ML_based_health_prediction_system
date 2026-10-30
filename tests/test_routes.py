import pytest
import json
from app import app as flask_app
from src.module_registry import ModuleRegistry
from src.predict import Predictor
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"HealthAI Analytics" in response.data

def test_evaluation_page(client):
    """Test if evaluation page loads successfully."""
    response = client.get('/evaluation')
    assert response.status_code == 200
    assert b"Model Evaluation" in response.data

def test_about_page(client):
    """Test if about page loads successfully."""
    response = client.get('/about')
    assert response.status_code == 200
    assert b"HealthAI Analytics" in response.data

def test_prediction_pages(client):
    """Test that all 4 condition prediction pages render."""
    for module_id in ['blood_pressure', 'diabetes', 'anemia', 'obesity']:
        response = client.get(f'/predict/{module_id}')
        assert response.status_code == 200
        assert module_id.encode() in response.data or b"Screening" in response.data

def test_api_modules(client):
    """Test if API returns all 4 available modules with schemas."""
    response = client.get('/api/modules')
    assert response.status_code == 200
    data = json.loads(response.data)
    for expected in ['blood_pressure', 'diabetes', 'anemia', 'obesity']:
        assert expected in data
        assert 'schema' in data[expected]
        assert 'name' in data[expected]

def test_module_schema(client):
    """Test fetching schema for specific module."""
    response = client.get('/api/modules/diabetes/schema')
    assert response.status_code == 200
    schema = json.loads(response.data)
    assert 'glucose' in str(schema).lower()

def test_invalid_module_schema(client):
    """Test if invalid module ID returns 404."""
    response = client.get('/api/modules/invalid_module_123/schema')
    assert response.status_code == 404

def test_valid_diabetes_prediction(client):
    """Test valid diabetes screening prediction."""
    payload = {
        'age': 50,
        'bmi': 27.5,
        'HbA1c_level': 6.2,
        'blood_glucose_level': 115.0
    }
    response = client.post('/api/predict/diabetes',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'primary_result' in data
    assert 'risk_tier' in data
    assert 'contributing_factors' in data
    assert 'model_comparisons' in data
    assert 'logistic_regression' in data['model_comparisons']
    assert 'decision_tree' in data['model_comparisons']
    assert 'random_forest' in data['model_comparisons']

def test_valid_anemia_prediction(client):
    """Test valid anemia screening prediction."""
    payload = {
        'Gender': 'Female',
        'Hemoglobin': 10.5,
        'MCH': 24.0,
        'MCHC': 29.0,
        'MCV': 75.0
    }
    response = client.post('/api/predict/anemia',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['module_id'] == 'anemia'
    assert data['primary_result']['score'] >= 0.0

def test_valid_obesity_prediction(client):
    """Test valid obesity classification prediction."""
    payload = {
        'Gender': 'Male',
        'Age': 26,
        'Height': 1.80,
        'Weight': 75.0,
        'family_history_with_overweight': 'no',
        'FAVC': 'no',
        'MTRANS': 'Public_Transportation'
    }
    response = client.post('/api/predict/obesity',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'Normal' in data['primary_result']['human_label'] or 'Weight' in data['primary_result']['human_label']

def test_input_range_validation_failure(client):
    """Test that physiologically impossible inputs trigger HTTP 400 with actionable error messages."""
    payload = {
        'age': 300,  # Exceeds max 100
        'bmi': -10.0, # Below min 10
        'HbA1c_level': 5.5,
        'blood_glucose_level': 100.0
    }
    response = client.post('/api/predict/diabetes',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Clinical input validation failed' in data['error']
    assert len(data['details']) >= 2

def test_evaluation_api(client):
    """Test evaluation report API returns verified metrics and dataset metadata."""
    response = client.get('/api/evaluation/diabetes')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'evaluation' in data
    assert 'logistic_regression' in data['evaluation']
    assert 'random_forest' in data['evaluation']
    assert 'dataset_info' in data
    assert data['dataset_info']['total_samples'] > 0

def test_history_api(client):
    """Test privacy-first screening history recording and clearing."""
    # Fetch history
    res1 = client.get('/api/history')
    assert res1.status_code == 200

    # Clear history
    res2 = client.delete('/api/history')
    assert res2.status_code == 200
    data2 = json.loads(res2.data)
    assert len(data2['history']) == 0
