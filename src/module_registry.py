class ModuleRegistry:
    """Central registry for all health prediction modules with validation schemas and clinical metadata."""

    MODULES = {
        'blood_pressure': {
            'name': 'Blood Pressure / Hypertension',
            'category': 'Cardiovascular',
            'description': 'Early screening for hypertension risk based on patient physiological and genetic factors.',
            'target_name': 'Hypertension Risk',
            'schema': {
                'Sex': {
                    'type': 'select',
                    'options': ['Male', 'Female'],
                    'label': 'Biological Sex',
                    'default': 'Male',
                    'tooltip': 'Biological sex at birth'
                },
                'Age': {
                    'type': 'number',
                    'label': 'Patient Age',
                    'unit': 'years',
                    'min': 18,
                    'max': 100,
                    'step': 1,
                    'default': 45,
                    'tooltip': 'Age between 18 and 100 years'
                },
                'BMI': {
                    'type': 'number',
                    'label': 'Body Mass Index (BMI)',
                    'unit': 'kg/m²',
                    'min': 10.0,
                    'max': 60.0,
                    'step': 0.1,
                    'default': 25.5,
                    'tooltip': 'Weight (kg) divided by height squared (m²). Normal is 18.5 - 24.9'
                },
                'Level_of_Hemoglobin': {
                    'type': 'number',
                    'label': 'Hemoglobin Level',
                    'unit': 'g/dL',
                    'min': 5.0,
                    'max': 22.0,
                    'step': 0.1,
                    'default': 13.8,
                    'tooltip': 'Normal range: 12.0 - 17.5 g/dL'
                },
                'Genetic_Pedigree_Coefficient': {
                    'type': 'number',
                    'label': 'Genetic Pedigree Coefficient',
                    'unit': '0.0 - 1.0',
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'default': 0.35,
                    'tooltip': 'Familial predisposition coefficient (0 = no known family history, 1 = strong lineage)'
                },
                'Chronic_kidney_disease': {
                    'type': 'select',
                    'options': ['0', '1'],
                    'option_labels': {'0': 'No History (0)', '1': 'Diagnosed (1)'},
                    'label': 'Chronic Kidney Disease History',
                    'default': '0',
                    'tooltip': 'Documented diagnosis of chronic renal impairment'
                }
            },
            'sample_cases': {
                'normal': {
                    'label': 'Typical Low Risk Profile',
                    'data': {'Sex': 'Female', 'Age': 32, 'BMI': 22.0, 'Level_of_Hemoglobin': 13.2, 'Genetic_Pedigree_Coefficient': 0.12, 'Chronic_kidney_disease': '0'}
                },
                'elevated': {
                    'label': 'Elevated Hypertension Risk Profile',
                    'data': {'Sex': 'Male', 'Age': 58, 'BMI': 31.5, 'Level_of_Hemoglobin': 16.4, 'Genetic_Pedigree_Coefficient': 0.78, 'Chronic_kidney_disease': '1'}
                }
            }
        },
        'diabetes': {
            'name': 'Diabetes / Blood Sugar',
            'category': 'Metabolic',
            'description': 'Early screening for Type-2 diabetes risk based on key clinical and biochemical indicators.',
            'target_name': 'Diabetes Risk',
            'schema': {
                'age': {
                    'type': 'number',
                    'label': 'Patient Age',
                    'unit': 'years',
                    'min': 1,
                    'max': 100,
                    'step': 1,
                    'default': 45,
                    'tooltip': 'Patient age in years'
                },
                'bmi': {
                    'type': 'number',
                    'label': 'Body Mass Index (BMI)',
                    'unit': 'kg/m²',
                    'min': 10.0,
                    'max': 70.0,
                    'step': 0.1,
                    'default': 24.8,
                    'tooltip': 'Normal: 18.5 - 24.9 kg/m²'
                },
                'HbA1c_level': {
                    'type': 'number',
                    'label': 'Glycated Hemoglobin (HbA1c)',
                    'unit': '%',
                    'min': 3.5,
                    'max': 16.0,
                    'step': 0.1,
                    'default': 5.4,
                    'tooltip': 'Normal: <5.7%, Prediabetes: 5.7 - 6.4%, Diabetic: ≥6.5%'
                },
                'blood_glucose_level': {
                    'type': 'number',
                    'label': 'Fasting Blood Glucose',
                    'unit': 'mg/dL',
                    'min': 40.0,
                    'max': 400.0,
                    'step': 1.0,
                    'default': 95.0,
                    'tooltip': 'Normal fasting: 70 - 99 mg/dL, Impaired: 100 - 125 mg/dL, Diabetic: ≥126 mg/dL'
                }
            },
            'sample_cases': {
                'normal': {
                    'label': 'Normal Glycemic Profile',
                    'data': {'age': 35, 'bmi': 23.0, 'HbA1c_level': 5.2, 'blood_glucose_level': 90.0}
                },
                'elevated': {
                    'label': 'Elevated Glycemic Risk Profile',
                    'data': {'age': 54, 'bmi': 32.4, 'HbA1c_level': 7.6, 'blood_glucose_level': 175.0}
                }
            }
        },
        'anemia': {
            'name': 'Anemia Risk',
            'category': 'Hematology',
            'description': 'Hematological screening for anemia based on hemoglobin concentration and red blood cell indices.',
            'target_name': 'Anemia Risk',
            'schema': {
                'Gender': {
                    'type': 'select',
                    'options': ['Male', 'Female'],
                    'label': 'Biological Sex',
                    'default': 'Female',
                    'tooltip': 'Biological sex (hemoglobin cutoffs differ between males and females)'
                },
                'Hemoglobin': {
                    'type': 'number',
                    'label': 'Hemoglobin Concentration',
                    'unit': 'g/dL',
                    'min': 4.0,
                    'max': 25.0,
                    'step': 0.1,
                    'default': 13.5,
                    'tooltip': 'Normal: ≥12.0 g/dL (Females), ≥13.5 g/dL (Males)'
                },
                'MCH': {
                    'type': 'number',
                    'label': 'Mean Corpuscular Hemoglobin (MCH)',
                    'unit': 'pg',
                    'min': 10.0,
                    'max': 50.0,
                    'step': 0.1,
                    'default': 29.5,
                    'tooltip': 'Average mass of hemoglobin per red cell. Normal: 27 - 33 pg'
                },
                'MCHC': {
                    'type': 'number',
                    'label': 'Mean Corpuscular Hemoglobin Concentration (MCHC)',
                    'unit': 'g/dL',
                    'min': 15.0,
                    'max': 45.0,
                    'step': 0.1,
                    'default': 33.2,
                    'tooltip': 'Average concentration of hemoglobin in a given volume of red cells. Normal: 32 - 36 g/dL'
                },
                'MCV': {
                    'type': 'number',
                    'label': 'Mean Corpuscular Volume (MCV)',
                    'unit': 'fL',
                    'min': 50.0,
                    'max': 130.0,
                    'step': 0.1,
                    'default': 88.0,
                    'tooltip': 'Average volume of red blood cells. Normal: 80 - 100 fL'
                }
            },
            'sample_cases': {
                'normal': {
                    'label': 'Normal Hematological Profile',
                    'data': {'Gender': 'Female', 'Hemoglobin': 14.1, 'MCH': 30.5, 'MCHC': 33.8, 'MCV': 90.2}
                },
                'elevated': {
                    'label': 'Microcytic Anemia Suspected',
                    'data': {'Gender': 'Female', 'Hemoglobin': 9.2, 'MCH': 22.0, 'MCHC': 28.5, 'MCV': 71.0}
                }
            }
        },
        'obesity': {
            'name': 'Obesity & Body Composition',
            'category': 'Nutrition & Anthropometry',
            'description': 'Multi-class body mass classification based on anthropometrics and physical activity habits.',
            'target_name': 'Obesity Classification Level',
            'schema': {
                'Gender': {
                    'type': 'select',
                    'options': ['Male', 'Female'],
                    'label': 'Biological Sex',
                    'default': 'Male',
                    'tooltip': 'Biological sex'
                },
                'Age': {
                    'type': 'number',
                    'label': 'Age',
                    'unit': 'years',
                    'min': 14,
                    'max': 100,
                    'step': 1,
                    'default': 25,
                    'tooltip': 'Age in years'
                },
                'Height': {
                    'type': 'number',
                    'label': 'Height',
                    'unit': 'meters',
                    'min': 1.0,
                    'max': 2.5,
                    'step': 0.01,
                    'default': 1.72,
                    'tooltip': 'Height in meters (e.g., 1.75 m for 175 cm)'
                },
                'Weight': {
                    'type': 'number',
                    'label': 'Weight',
                    'unit': 'kg',
                    'min': 30.0,
                    'max': 250.0,
                    'step': 0.5,
                    'default': 70.0,
                    'tooltip': 'Weight in kilograms'
                },
                'family_history_with_overweight': {
                    'type': 'select',
                    'options': ['yes', 'no'],
                    'option_labels': {'yes': 'Yes (Family History)', 'no': 'No'},
                    'label': 'Family History of Overweight',
                    'default': 'no',
                    'tooltip': 'Has a family member suffered from overweight or obesity'
                },
                'FAVC': {
                    'type': 'select',
                    'options': ['yes', 'no'],
                    'option_labels': {'yes': 'Yes (Frequent)', 'no': 'No'},
                    'label': 'Frequent High Caloric Food Consumption',
                    'default': 'yes',
                    'tooltip': 'Do you frequently eat high-calorie meals'
                },
                'MTRANS': {
                    'type': 'select',
                    'options': ['Public_Transportation', 'Private_Transportation', 'Walking', 'Biking', 'Motorbike'],
                    'option_labels': {
                        'Public_Transportation': 'Public Transit',
                        'Private_Transportation': 'Automobile',
                        'Walking': 'Walking',
                        'Biking': 'Bicycle',
                        'Motorbike': 'Motorbike'
                    },
                    'label': 'Primary Mode of Transportation',
                    'default': 'Public_Transportation',
                    'tooltip': 'Primary commute method'
                }
            },
            'sample_cases': {
                'normal': {
                    'label': 'Normal Weight Profile',
                    'data': {'Gender': 'Male', 'Age': 24, 'Height': 1.78, 'Weight': 72.0, 'family_history_with_overweight': 'no', 'FAVC': 'no', 'MTRANS': 'Walking'}
                },
                'elevated': {
                    'label': 'Elevated Body Mass Profile',
                    'data': {'Gender': 'Female', 'Age': 32, 'Height': 1.62, 'Weight': 95.0, 'family_history_with_overweight': 'yes', 'FAVC': 'yes', 'MTRANS': 'Private_Transportation'}
                }
            }
        }
    }

    @classmethod
    def get_module(cls, module_id):
        return cls.MODULES.get(module_id)

    @classmethod
    def list_modules(cls):
        return cls.MODULES
