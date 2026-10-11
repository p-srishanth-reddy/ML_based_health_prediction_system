import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic validation of health datasets."""

    DATA_CONFIG = {
        'blood_pressure': {
            'path': 'data/blood_pressure/blood_press.csv',
            'expected_cols': ['Blood_Pressure_Abnormality', 'Age', 'Sex', 'BMI', 'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Chronic_kidney_disease'],
            'target': 'Blood_Pressure_Abnormality'
        },
        'diabetes': {
            'path': 'data/diabetes/diabetes.csv',
            'expected_cols': ['bmi', 'HbA1c_level', 'blood_glucose_level', 'age', 'diabetes'],
            'target': 'diabetes'
        },
        'anemia': {
            'path': 'data/anemia/anemia.csv',
            'expected_cols': ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Result'],
            'target': 'Result'
        },
        'obesity': {
            'path': 'data/obesity/obesity.csv',
            'expected_cols': ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'],
            'target': 'NObeyesdad'
        }
    }

    @staticmethod
    def load_dataset(module_id):
        """Loads a dataset based on the module ID."""
        if module_id not in DataLoader.DATA_CONFIG:
            raise ValueError(f"Unknown module ID: {module_id}")

        config = DataLoader.DATA_CONFIG[module_id]
        path = config['path']

        if not os.path.exists(path):
            logger.error(f"Dataset file not found at {path}")
            raise FileNotFoundError(f"Dataset file missing: {path}. Please check data/README.md")

        try:
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {module_id} dataset from {path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {module_id} dataset: {e}")
            raise e

    @staticmethod
    def validate_schema(df, module_id):
        """Validates that the loaded dataframe contains the expected columns."""
        config = DataLoader.DATA_CONFIG[module_id]
        expected = config['expected_cols']
        actual = df.columns.tolist()

        missing = [col for col in expected if col not in actual]
        if missing:
            logger.warning(f"Module {module_id} is missing expected columns: {missing}")
            return False, missing

        return True, []
