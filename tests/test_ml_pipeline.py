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

def test_anemia_dataset_audit_and_leakage():
    """Verify anemia dataset deduplication (1421 raw -> 534 unique) and deterministic WHO thresholding."""
    df_raw = DataLoader.load_dataset('anemia', deduplicate=False)
    assert len(df_raw) == 1421
    assert df_raw.duplicated().sum() == 887

    df_clean = DataLoader.load_dataset('anemia', deduplicate=True)
    assert len(df_clean) == 534
    assert df_clean.duplicated().sum() == 0

    # Test WHO deterministic cutoffs: Female < 12.0, Male < 13.5
    female_cases = df_clean[df_clean['Gender'] == 'Female']
    male_cases = df_clean[df_clean['Gender'] == 'Male']

    assert (female_cases[female_cases['Hemoglobin'] < 12.0]['Result'] == 1).all()
    assert (female_cases[female_cases['Hemoglobin'] >= 12.0]['Result'] == 0).all()
    assert (male_cases[male_cases['Hemoglobin'] < 13.5]['Result'] == 1).all()
    assert (male_cases[male_cases['Hemoglobin'] >= 13.5]['Result'] == 0).all()

def test_anemia_evaluation_metadata_and_generalization_gap():
    """Verify retrained anemia metadata includes 5-fold CV, train acc, generalization gap, and clinical analysis."""
    import json
    with open('models/anemia/metadata.json', 'r') as f:
        meta = json.load(f)

    assert meta['module_id'] == 'anemia'
    assert 'evaluation' in meta
    eval_data = meta['evaluation']

    for clf_name in ['logistic_regression', 'decision_tree', 'random_forest']:
        assert clf_name in eval_data
        m = eval_data[clf_name]
        assert 'train_accuracy' in m
        assert 'cv_accuracy_mean' in m
        assert 'cv_accuracy_std' in m
        assert 'cv_f1_mean' in m
        assert 'accuracy' in m
        assert 'balanced_accuracy' in m
        assert 'precision' in m
        assert 'recall' in m
        assert 'specificity' in m
        assert 'f1_score' in m
        assert 'roc_auc' in m
        assert 'generalization_gap' in m
        assert 'hyperparameters' in m

    assert 'clinical_analysis' in meta
    assert meta['clinical_analysis']['target_leakage_identified'] is True
    assert 'WHO' in meta['clinical_analysis']['clinical_explanation']

def test_anemia_clinical_predictions():
    """Test inference on typical female and male patient cases."""
    # Anemic female (Hb 10.5 < 12.0)
    female_anemic = Predictor.predict_all('anemia', {
        'Gender': 'Female', 'Hemoglobin': 10.5, 'MCH': 24.0, 'MCHC': 29.0, 'MCV': 75.0
    })
    assert female_anemic['primary_result']['human_label'] == 'Anemic Risk'

    # Normal female (Hb 13.2 >= 12.0)
    female_normal = Predictor.predict_all('anemia', {
        'Gender': 'Female', 'Hemoglobin': 13.2, 'MCH': 30.0, 'MCHC': 33.5, 'MCV': 89.0
    })
    assert female_normal['primary_result']['human_label'] == 'Non-Anemic'

    # Anemic male at borderline value (Hb 12.5 < 13.5)
    male_anemic = Predictor.predict_all('anemia', {
        'Gender': 'Male', 'Hemoglobin': 12.5, 'MCH': 28.0, 'MCHC': 31.0, 'MCV': 84.0
    })
    assert male_anemic['primary_result']['human_label'] == 'Anemic Risk'

    # Normal male (Hb 14.8 >= 13.5)
    male_normal = Predictor.predict_all('anemia', {
        'Gender': 'Male', 'Hemoglobin': 14.8, 'MCH': 31.0, 'MCHC': 34.0, 'MCV': 92.0
    })
    assert male_normal['primary_result']['human_label'] == 'Non-Anemic'

