import pandas as pd
from src.data_loader import DataLoader

class ModuleRegistry:
    """Central registry for all health prediction modules."""

    MODULES = {
        'blood_pressure': {
            'name': 'Blood Pressure / Hypertension',
            'description': 'Screening for hypertension risk based on patient clinical metrics.',
            'schema': {
                'Sex': {'type': 'select', 'options': ['Male', 'Female'], 'label': 'Gender'},
                'Age': {'type': 'number', 'label': 'Age (years)'},
                'BMI': {'type': 'number', 'label': 'BMI'},
                'Level_of_Hemoglobin': {'type': 'number', 'label': 'Hemoglobin Level'},
                'Genetic_Pedigree_Coefficient': {'type': 'number', 'label': 'Genetic Coefficient'},
                'Chronic_kidney_disease': {'type': 'select', 'options': ['0', '1'], 'label': 'Chronic Kidney Disease (0=No, 1=Yes)'},
            }
        },
        'diabetes': {
            'name': 'Diabetes / Sugar',
            'description': 'Screening for diabetes risk based on clinical measurements.',
            'schema': {
                'bmi': {'type': 'number', 'label': 'BMI'},
                'HbA1c_level': {'type': 'number', 'label': 'HbA1c Level'},
                'blood_glucose_level': {'type': 'number', 'label': 'Blood Glucose (mg/dL)'},
                'age': {'type': 'number', 'label': 'Age (years)'},
            }
        },
        'anemia': {
            'name': 'Anemia Risk',
            'description': 'Screening for anemia risk based on hemoglobin and RBC indices.',
            'schema': {
                'Gender': {'type': 'select', 'options': ['Male', 'Female'], 'label': 'Gender'},
                'Hemoglobin': {'type': 'number', 'label': 'Hemoglobin (g/dL)'},
                'MCHC': {'type': 'number', 'label': 'MCHC (g/dL)'},
                'MCV': {'type': 'number', 'label': 'MCV (fL)'},
                'MCH': {'type': 'number', 'label': 'MCH (pg)'},
            }
        },
        'obesity': {
            'name': 'Obesity Level',
            'description': 'Classification of obesity levels based on key physical metrics.',
            'schema': {
                'Gender': {'type': 'select', 'options': ['Male', 'Female'], 'label': 'Gender'},
                'Age': {'type': 'number', 'label': 'Age (years)'},
                'Height': {'type': 'number', 'label': 'Height (m)'},
                'Weight': {'type': 'number', 'label': 'Weight (kg)'},
                'family_history_with_overweight': {'type': 'select', 'options': ['yes', 'no'], 'label': 'Family history of overweight'},
                'FAVC': {'type': 'select', 'options': ['yes', 'no'], 'label': 'Frequent high-calorie food consumption'},
                'MTRANS': {'type': 'select', 'options': ['Public_Transportation', 'Private_Transportation', 'Walking', 'Biking'], 'label': 'Transportation method'},
            }
        }
    }

    @classmethod
    def get_module(cls, module_id):
        return cls.MODULES.get(module_id)

    @classmethod
    def list_modules(cls):
        return cls.MODULES
