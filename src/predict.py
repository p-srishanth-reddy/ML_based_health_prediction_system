import joblib
import json
import os
import numpy as np
import pandas as pd
from src.module_registry import ModuleRegistry

class Predictor:
    """Handles loading models, computing multi-model predictions, and generating explainability metrics."""

    LABEL_NAMES = {
        'anemia': {'0': 'Non-Anemic', '1': 'Anemic Risk'},
        'blood_pressure': {'0': 'Normal BP Risk', '1': 'Elevated Hypertension Risk'},
        'diabetes': {'0': 'Non-Diabetic', '1': 'Elevated Diabetes Risk'},
        'obesity': {
            'Insufficient_Weight': 'Underweight',
            'Normal_Weight': 'Normal Weight',
            'Overweight_Level_I': 'Overweight (Class I)',
            'Overweight_Level_II': 'Overweight (Class II)',
            'Obesity_Type_I': 'Obesity (Class I)',
            'Obesity_Type_II': 'Obesity (Class II)',
            'Obesity_Type_III': 'Obesity (Class III - Severe)'
        }
    }

    # Clinical reference baselines used to evaluate patient-level feature deviations
    BASELINES = {
        'diabetes': {
            'HbA1c_level': {'normal': 5.4, 'weight': 0.45, 'high_bad': True, 'unit': '%', 'name': 'HbA1c Level'},
            'blood_glucose_level': {'normal': 95.0, 'weight': 0.35, 'high_bad': True, 'unit': 'mg/dL', 'name': 'Blood Glucose'},
            'bmi': {'normal': 22.5, 'weight': 0.10, 'high_bad': True, 'unit': 'kg/m²', 'name': 'Body Mass Index'},
            'age': {'normal': 35.0, 'weight': 0.10, 'high_bad': True, 'unit': 'years', 'name': 'Patient Age'}
        },
        'blood_pressure': {
            'Genetic_Pedigree_Coefficient': {'normal': 0.20, 'weight': 0.35, 'high_bad': True, 'unit': '', 'name': 'Genetic Pedigree'},
            'Level_of_Hemoglobin': {'normal': 13.5, 'weight': 0.30, 'high_bad': True, 'unit': 'g/dL', 'name': 'Hemoglobin Level'},
            'BMI': {'normal': 23.0, 'weight': 0.15, 'high_bad': True, 'unit': 'kg/m²', 'name': 'BMI'},
            'Age': {'normal': 35.0, 'weight': 0.10, 'high_bad': True, 'unit': 'years', 'name': 'Age'},
            'Chronic_kidney_disease': {'normal': '0', 'weight': 0.10, 'high_bad': True, 'unit': '', 'name': 'Chronic Kidney Disease'}
        },
        'anemia': {
            'Hemoglobin': {'normal': 14.0, 'weight': 0.70, 'high_bad': False, 'unit': 'g/dL', 'name': 'Hemoglobin Level'},
            'MCV': {'normal': 90.0, 'weight': 0.15, 'high_bad': False, 'unit': 'fL', 'name': 'Mean Corpuscular Volume'},
            'MCH': {'normal': 30.0, 'weight': 0.10, 'high_bad': False, 'unit': 'pg', 'name': 'Mean Corpuscular Hemoglobin'},
            'MCHC': {'normal': 33.5, 'weight': 0.05, 'high_bad': False, 'unit': 'g/dL', 'name': 'MCHC'}
        },
        'obesity': {
            'Weight': {'normal': 65.0, 'weight': 0.50, 'high_bad': True, 'unit': 'kg', 'name': 'Weight'},
            'Height': {'normal': 1.72, 'weight': 0.20, 'high_bad': False, 'unit': 'm', 'name': 'Height'},
            'Age': {'normal': 25.0, 'weight': 0.15, 'high_bad': True, 'unit': 'years', 'name': 'Age'},
            'FAVC': {'normal': 'no', 'weight': 0.15, 'high_bad': True, 'unit': '', 'name': 'High-Calorie Diet'}
        }
    }

    @staticmethod
    def predict(module_id, input_data, model_type='random_forest'):
        """Makes a single-model prediction for backwards compatibility."""
        model_dir = f"models/{module_id}"
        model_path = f"{model_dir}/{model_type}.joblib"
        meta_path = f"{model_dir}/metadata.json"

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model or metadata missing for {module_id} ({model_type}).")

        pipeline = joblib.load(model_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        features = meta['features']

        # Format input data dictionary
        clean_input = {}
        for k in features:
            v = input_data.get(k)
            if v is not None and v != '':
                clean_input[k] = v
            else:
                clean_input[k] = np.nan

        df_input = pd.DataFrame([clean_input])[features]

        # Make prediction
        prediction = pipeline.predict(df_input)[0]
        probabilities = pipeline.predict_proba(df_input)[0]

        # Label mapping
        target_mapping = meta['target_mapping']
        label = None
        for k, v in target_mapping.items():
            if v == prediction:
                label = k
                break
        if label is None:
            label = str(prediction)

        # Calculate score
        is_binary = len(meta['classes']) == 2
        if is_binary:
            pos_idx = 1
            for idx, c in enumerate(meta['classes']):
                if str(c).lower() in ['1', 'yes', 'anemic', 'diabetic', 'hypertensive', 'abnormal']:
                    pos_idx = idx
                    break
            raw_score = probabilities[pos_idx] * 100
            score = max(0.1, min(99.9, float(raw_score)))
        else:
            pred_idx = int(prediction) if str(prediction).isdigit() else 0
            score = max(0.1, min(99.9, float(probabilities[pred_idx] * 100)))

        human_label = Predictor.LABEL_NAMES.get(module_id, {}).get(label, label)

        return {
            'prediction': label,
            'human_label': human_label,
            'score': round(float(score), 2),
            'probabilities': [round(float(p), 4) for p in probabilities],
            'model_type': model_type,
            'module_id': module_id
        }

    @classmethod
    def predict_all(cls, module_id, input_data):
        """Runs predictions across Logistic Regression, Decision Tree, and Random Forest models with explainability."""
        meta_path = f"models/{module_id}/metadata.json"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata missing for module {module_id}.")

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        primary_model = meta.get('primary_model', 'random_forest')

        # Run inference for all available models
        model_results = {}
        for m_name in ['logistic_regression', 'decision_tree', 'random_forest']:
            m_path = f"models/{module_id}/{m_name}.joblib"
            if os.path.exists(m_path):
                model_results[m_name] = cls.predict(module_id, input_data, model_type=m_name)

        if not model_results:
            raise FileNotFoundError(f"No trained models found for {module_id}")

        primary_result = model_results.get(primary_model, next(iter(model_results.values())))
        is_binary = len(meta['classes']) == 2
        score = primary_result['score']

        # Determine clinical risk tier
        if is_binary:
            if score < 25.0:
                risk_tier = 'Low Risk'
                risk_badge = 'success'
                recommendation = "Metrics fall within low statistical risk ranges. Maintain regular annual health screenings and a balanced lifestyle."
            elif score < 50.0:
                risk_tier = 'Moderate / Borderline Risk'
                risk_badge = 'warning'
                recommendation = "Some clinical parameters are borderline. Consider discussing lifestyle modifications or follow-up laboratory screening with your physician."
            elif score < 75.0:
                risk_tier = 'Elevated Risk'
                risk_badge = 'warning'
                recommendation = "Multiple indicators show elevated correlation with condition risk in the training cohort. A consultation with a qualified medical professional is advised for formal diagnostic testing."
            else:
                risk_tier = 'High Risk'
                risk_badge = 'danger'
                recommendation = "Key markers indicate significant statistical risk patterns. Clinical evaluation and confirmatory medical diagnostics are strongly recommended."
        else:
            # Obesity multi-class
            risk_tier = primary_result['human_label']
            if 'Normal' in risk_tier:
                risk_badge = 'success'
                recommendation = "Body mass parameters align with the healthy reference range. Continue balanced nutrition and regular physical activity."
            elif 'Underweight' in risk_tier:
                risk_badge = 'info'
                recommendation = "Body mass index is below the standard reference range. Consider dietary evaluation with a certified nutritionist."
            elif 'Overweight' in risk_tier:
                risk_badge = 'warning'
                recommendation = "Body mass index indicates overweight status. Cardiovascular exercise and nutritional optimization are beneficial."
            else:
                risk_badge = 'danger'
                recommendation = "Body mass indicators indicate obesity range. Medical supervision and targeted lifestyle intervention are recommended."

        # Compute individual patient feature contributions
        contributions = cls._compute_feature_contributions(module_id, input_data)

        return {
            'module_id': module_id,
            'primary_model': primary_model,
            'primary_result': primary_result,
            'model_comparisons': model_results,
            'risk_tier': risk_tier,
            'risk_badge': risk_badge,
            'recommendation': recommendation,
            'contributing_factors': contributions,
            'evaluation_metrics': meta.get('evaluation', {}),
            'explainability_notice': (
                "Feature influence indicates statistical association in the clinical training dataset, "
                "not direct medical causation. Model probability is an uncalibrated algorithmic score."
            )
        }

    @classmethod
    def _compute_feature_contributions(cls, module_id, input_data):
        """Calculates patient-specific feature influences compared against clinical reference baselines."""
        factors = []
        baselines = cls.BASELINES.get(module_id, {})

        for feature, spec in baselines.items():
            val = input_data.get(feature)
            if val is None or val == '':
                continue

            ref_normal = spec['normal']
            weight = spec['weight']
            high_bad = spec['high_bad']
            display_name = spec['name']
            unit = spec['unit']

            try:
                # Numeric comparison
                if isinstance(ref_normal, (int, float)):
                    num_val = float(val)
                    diff = num_val - ref_normal
                    pct_diff = (diff / ref_normal) if ref_normal != 0 else 0

                    if high_bad:
                        # Values higher than normal increase risk
                        impact = pct_diff * weight * 100
                    else:
                        # Values lower than normal increase risk (e.g., hemoglobin)
                        impact = -pct_diff * weight * 100

                    direction = 'increases_risk' if impact > 2 else ('reduces_risk' if impact < -2 else 'neutral')
                    display_val = f"{num_val} {unit}".strip()

                    factors.append({
                        'feature': feature,
                        'name': display_name,
                        'value': display_val,
                        'reference': f"Ref: ~{ref_normal} {unit}".strip(),
                        'impact_score': round(abs(impact), 1),
                        'direction': direction,
                        'description': (
                            f"{display_name} is higher than reference ({ref_normal} {unit})"
                            if diff > 0 and high_bad else (
                                f"{display_name} is lower than reference ({ref_normal} {unit})"
                                if diff < 0 and not high_bad else f"{display_name} is within healthy range"
                            )
                        )
                    })
                else:
                    # Categorical comparison
                    is_elevated = (str(val).lower() != str(ref_normal).lower())
                    impact = 40.0 * weight if is_elevated else -15.0 * weight
                    direction = 'increases_risk' if is_elevated else 'reduces_risk'

                    factors.append({
                        'feature': feature,
                        'name': display_name,
                        'value': str(val),
                        'reference': f"Ref: {ref_normal}",
                        'impact_score': round(abs(impact), 1),
                        'direction': direction,
                        'description': f"{display_name} contributes to elevated screening probability" if is_elevated else f"{display_name} reduces risk"
                    })
            except Exception:
                continue

        # Sort factors by impact score
        factors.sort(key=lambda x: x['impact_score'], reverse=True)
        return factors
