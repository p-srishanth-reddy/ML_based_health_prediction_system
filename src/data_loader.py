import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading, cleaning, deduplication, and schema validation of health datasets."""

    DATA_CONFIG = {
        'blood_pressure': {
            'path': 'data/blood_pressure/blood_press.csv',
            'expected_cols': ['Age', 'Sex', 'BMI', 'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient', 'Chronic_kidney_disease', 'Blood_Pressure_Abnormality'],
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
            'expected_cols': ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'MTRANS', 'NObeyesdad'],
            'target': 'NObeyesdad'
        }
    }

    @staticmethod
    def load_dataset(module_id, deduplicate=True):
        """Loads and cleans a dataset based on the module ID."""
        if module_id not in DataLoader.DATA_CONFIG:
            raise ValueError(f"Unknown module ID: {module_id}")

        config = DataLoader.DATA_CONFIG[module_id]
        path = config['path']

        if not os.path.exists(path):
            logger.error(f"Dataset file not found at {path}")
            raise FileNotFoundError(f"Dataset file missing: {path}. Please check data/README.md")

        try:
            df = pd.read_csv(path)
            raw_len = len(df)

            # Standardize column mappings and categorical representations
            if module_id == 'anemia':
                # Anemia dataset: 0 = Male, 1 = Female
                if 'Gender' in df.columns and df['Gender'].dtype != 'object':
                    df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'}).fillna('Male')

            elif module_id == 'blood_pressure':
                # Blood pressure dataset: Sex 0 = Male, 1 = Female
                if 'Sex' in df.columns and df['Sex'].dtype != 'object':
                    df['Sex'] = df['Sex'].map({0: 'Male', 1: 'Female'}).fillna('Male')
                if 'Chronic_kidney_disease' in df.columns:
                    df['Chronic_kidney_disease'] = df['Chronic_kidney_disease'].astype(str)

            # Keep only expected columns to ensure pipeline consistency
            expected = config['expected_cols']
            existing_expected = [c for c in expected if c in df.columns]
            df = df[existing_expected]

            # Deduplicate across the expected feature set to strictly prevent test-set leakage
            if deduplicate:
                df = df.drop_duplicates().reset_index(drop=True)
                logger.info(f"Loaded {module_id}: {raw_len} raw rows -> {len(df)} deduplicated rows")

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
