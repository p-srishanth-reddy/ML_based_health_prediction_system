import joblib
import json
import os
import numpy as np
import pandas as pd
from src.data_loader import DataLoader

class Predictor:
    """Handles loading models and making predictions."""

    @staticmethod
    def predict(module_id, input_data, model_type='random_forest'):
        """
        Makes a prediction for a given module and input data.
        input_data: dict of {feature_name: value}
        """
        model_dir = f"models/{module_id}"
        model_path = f"{model_dir}/{model_type}.joblib"
        meta_path = f"{model_dir}/metadata.json"

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model or metadata missing for {module_id}. Please train the models first.")

        # Load pipeline and metadata
        pipeline = joblib.load(model_path)
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Prepare input dataframe
        features = meta['features']
        # Ensure input data only contains the expected features in correct order
        df_input = pd.DataFrame([ {k: v for k, v in input_data.items() if k in features} ])

        # Add missing features with default values (though validation should prevent this)
        for col in features:
            if col not in df_input.columns:
                df_input[col] = np.nan

        df_input = df_input[features]

        # Predict
        prediction = pipeline.predict(df_input)[0]
        probabilities = pipeline.predict_proba(df_input)[0]

        # Map back to original label
        target_mapping = meta['target_mapping']
        label = None
        for k, v in target_mapping.items():
            if v == prediction:
                label = k
                break

        if label is None:
            label = str(prediction)

        # Calculate score for positive class (binary)
        score = 0.0
        if len(meta['classes']) == 2:
            # Find the index of the positive class
            pos_class_index = -1
            for idx, cls_name in enumerate(meta['classes']):
                if str(cls_name).lower() in ['1', 'yes', 'anemic', 'diabetic', 'hypertensive', 'abnormal']:
                    pos_class_index = idx
                    break

            if pos_class_index == -1:
                pos_class_index = 1 # Default to index 1

            # Calculate probability and enforce a minimum threshold of 0.1% to avoid 0.0%
            raw_score = probabilities[pos_class_index] * 100
            score = max(0.1, raw_score)
        else:
            # For multiclass (Obesity), use the probability of the predicted class
            # as the confidence score
            score = probabilities[prediction] * 100
            score = max(0.1, score)


        return {
            'prediction': label,
            'score': score,
            'probabilities': probabilities.tolist(),
            'model_type': model_type,
            'module_id': module_id
        }
