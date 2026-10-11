import argparse
import os
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor

def train_module(module_id):
    print(f"Training module: {module_id}...")

    # 1. Load Data
    try:
        df = DataLoader.load_dataset(module_id)
    except FileNotFoundError:
        print(f"Skipping {module_id}: Dataset not found.")
        return None

    # 2. Preprocess
    config = DataLoader.DATA_CONFIG[module_id]
    target_col = config['target']
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle Target encoding if categorical (e.g., Obesity)
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        target_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
    else:
        le = None
        target_mapping = {str(k): int(v) for k, v in zip(np.unique(y), np.unique(y))}

    preprocessor, features = Preprocessor.get_pipeline(module_id)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit preprocessor on training data only
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 3. Train Models
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    saved_models = {}

    for name, model in models.items():
        # We use class_weight='balanced' to fix the bias where healthy people are predicted as sick
        # We also use a very strict max_depth to prevent the model from 'over-guessing'
        depth = 3 if module_id == 'anemia' else 4

        if name == 'decision_tree':
            model = DecisionTreeClassifier(
                max_depth=depth,
                class_weight='balanced',
                random_state=42
            )
        elif name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=depth,
                class_weight='balanced',
                random_state=42
            )

        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)
        probs = model.predict_proba(X_test_proc)

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')

        # ROC-AUC
        try:
            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y_test, probs[:, 1])
            else:
                auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': confusion_matrix(y_test, preds).tolist()
        }

        # Save full pipeline (preprocessor + model)
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        model_dir = f"models/{module_id}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{name}.joblib"
        joblib.dump(full_pipeline, model_path)
        saved_models[name] = model_path

    # 4. Save Metadata & Evaluation
    metadata = {
        'module_id': module_id,
        'target_mapping': target_mapping,
        'features': features,
        'evaluation': results,
        'saved_models': saved_models,
        'classes': le.classes_.tolist() if le else list(target_mapping.keys())
    }

    meta_path = f"models/{module_id}/metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    return metadata

if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, help='Specific module to train')
    parser.add_argument('--all', action='store_true', help='Train all modules')
    args = parser.parse_args()

    if args.all:
        for module in DataLoader.DATA_CONFIG.keys():
            train_module(module)
    elif args.module:
        train_module(args.module)
    else:
        print("Please specify --module <id> or --all")
