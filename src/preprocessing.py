import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_loader import DataLoader

class Preprocessor:
    """Builds robust preprocessing pipelines for each health prediction module."""

    MODULE_COLUMNS = {
        'blood_pressure': {
            'num_cols': ['Age', 'BMI', 'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient'],
            'cat_cols': ['Sex', 'Chronic_kidney_disease']
        },
        'diabetes': {
            'num_cols': ['bmi', 'HbA1c_level', 'blood_glucose_level', 'age'],
            'cat_cols': []
        },
        'anemia': {
            'num_cols': ['Hemoglobin', 'MCH', 'MCHC', 'MCV'],
            'cat_cols': ['Gender']
        },
        'obesity': {
            'num_cols': ['Age', 'Height', 'Weight'],
            'cat_cols': ['Gender', 'family_history_with_overweight', 'FAVC', 'MTRANS']
        }
    }

    @staticmethod
    def get_pipeline(module_id):
        """Returns an unfitted ColumnTransformer and the ordered list of input features."""
        if module_id not in Preprocessor.MODULE_COLUMNS:
            raise ValueError(f"Unknown module ID: {module_id}")

        cols = Preprocessor.MODULE_COLUMNS[module_id]
        num_cols = cols['num_cols']
        cat_cols = cols['cat_cols']
        features = num_cols + cat_cols

        # Numerical pipeline: Median imputation + Standard Scaling
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        transformers = [('num', num_transformer, num_cols)]

        # Categorical pipeline: Most-frequent imputation + One-Hot Encoding
        if cat_cols:
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_transformer, cat_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        return preprocessor, features
