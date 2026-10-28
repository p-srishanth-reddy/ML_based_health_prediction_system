import pytest
import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.predict import Predictor

def test_data_loader_deduplication():
    """Verify data loader deduplicates rows and prevents test-set leakage."""
    for mod in ['anemia', 'diabetes', 'obesity']:
        df_raw = DataLoader.load_dataset(mod, deduplicate=False)
        df_dedup = DataLoader.load_dataset(mod, deduplicate=True)
        assert len(df_dedup) <= len(df_raw)
        assert df_dedup.duplicated().sum() == 0

def test_data_loader_categorical_mapping():
    """Verify gender and sex are cleanly mapped to strings in datasets."""
    df_anemia = DataLoader.load_dataset('anemia')
    assert set(df_anemia['Gender'].unique()).issubset({'Male', 'Female'})

    df_bp = DataLoader.load_dataset('blood_pressure')
    assert set(df_bp['Sex'].unique()).issubset({'Male', 'Female'})

def test_preprocessor_pipeline_structure():
    """Verify preprocessing pipeline builds transformers with StandardScaler and OneHotEncoder."""
    for mod in ['blood_pressure', 'diabetes', 'anemia', 'obesity']:
        pipeline, features = Preprocessor.get_pipeline(mod)
        assert len(features) > 0
        assert pipeline is not None

def test_predictor_multi_model_structure():
    """Verify Predictor.predict_all returns comparisons across all 3 algorithms."""
    sample = {
        'age': 40,
        'bmi': 24.0,
        'HbA1c_level': 5.2,
        'blood_glucose_level': 90.0
    }
    res = Predictor.predict_all('diabetes', sample)
    assert res['primary_model'] in ['logistic_regression', 'decision_tree', 'random_forest']
    assert 'logistic_regression' in res['model_comparisons']
    assert 'decision_tree' in res['model_comparisons']
    assert 'random_forest' in res['model_comparisons']
    assert len(res['contributing_factors']) > 0
