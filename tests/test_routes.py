import pytest
from app import app as flask_app
import json

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200

def test_api_modules(client):
    """Test if API returns available modules."""
    response = client.get('/api/modules')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'blood_pressure' in data
    assert 'diabetes' in data

def test_invalid_module_schema(client):
    """Test if invalid module ID returns 404."""
    response = client.get('/api/modules/invalid_id/schema')
    assert response.status_code == 404

def test_prediction_no_model(client):
    """Test prediction without trained models returns 500 or 400."""
    # We use a valid module but no models are actually on disk in test env usually
    response = client.post('/api/predict/blood_pressure',
                           data=json.dumps({'Age': 30, 'Gender': 'Male'}),
                           content_type='application/json')
    # Should be 500 (FileNotFoundError handled) or 400 (validation)
    assert response.status_code in [400, 500]
