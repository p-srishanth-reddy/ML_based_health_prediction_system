import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_loader import DataLoader

class Preprocessor:
    """Builds preprocessing pipelines for different health modules."""

    @staticmethod
    def get_pipeline(module_id):
        """Returns a fitted preprocessing pipeline for the given module."""
        config = DataLoader.DATA_CONFIG[module_id]
        target = config['target']
        features = [col for col in config['expected_cols'] if col != target]

        # Identify numerical and categorical columns
        # In a real scenario, we would inspect the loaded data.
        # Here we use reasonable defaults based on documentation.

        num_cols = []
        cat_cols = []

        if module_id == 'blood_pressure':
            cat_cols = ['Sex']
            num_cols = ['Age', 'BMI', 'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Chronic_kidney_disease']
        elif module_id == 'diabetes':
            num_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level', 'age']
        elif module_id == 'anemia':
            cat_cols = ['Gender']
            num_cols = ['Hemoglobin', 'MCH', 'MCHC', 'MCV']
        elif module_id == 'obesity':
            cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'MTRANS']
            num_cols = ['Age', 'Height', 'Weight']

        # Numerical pipeline: Impute missing with median
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])

        # Categorical pipeline: Impute missing with most frequent and OneHot encode
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_cols),
                ('cat', cat_transformer, cat_cols)
            ]
        )

        return preprocessor, features
