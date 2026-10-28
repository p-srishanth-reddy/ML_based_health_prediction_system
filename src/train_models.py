import argparse
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor

def train_module(module_id):
    print(f"\n==================================================")
    print(f"Training Module: {module_id.upper()}")
    print(f"==================================================")

    # 1. Load Clean, Deduplicated Data
    try:
        df = DataLoader.load_dataset(module_id, deduplicate=True)
    except FileNotFoundError as e:
        print(f"Skipping {module_id}: {e}")
        return None

    config = DataLoader.DATA_CONFIG[module_id]
    target_col = config['target']

    preprocessor, features = Preprocessor.get_pipeline(module_id)

    # Validate all expected features exist
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' not found in loaded dataframe for module '{module_id}'")

    X = df[features]
    y = df[target_col]

    # Target Encoding
    le = None
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        target_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
        classes = le.classes_.tolist()
    else:
        y_encoded = y.astype(int).values
        unique_classes = sorted(np.unique(y_encoded).tolist())
        target_mapping = {str(k): int(k) for k in unique_classes}
        classes = [str(k) for k in unique_classes]

    is_binary = len(classes) == 2

    # 2. Stratified Train-Test Split (80% Train, 20% Test) with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples | Classes: {classes}")

    # 3. Model Configurations & Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Algorithm candidate suite
    candidate_models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=4 if module_id != 'anemia' else 3,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=6 if module_id != 'anemia' else 4,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
    }

    results = {}
    saved_models = {}
    feature_importances = {}
    best_model_name = 'random_forest'
    best_f1 = -1.0

    model_dir = f"models/{module_id}"
    os.makedirs(model_dir, exist_ok=True)

    for name, clf in candidate_models.items():
        # Fit preprocessor on training data only to strictly prevent data leakage
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Cross-validation score on training partition
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
        mean_cv_f1 = float(np.mean(cv_scores))

        # Train on full training split
        pipeline.fit(X_train, y_train)

        # Evaluate on strictly held-out test split
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)

        acc = float(accuracy_score(y_test, preds))
        bal_acc = float(balanced_accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, preds, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, preds, average='weighted', zero_division=0))
        cm = confusion_matrix(y_test, preds).tolist()

        # Specificity & PR-AUC
        specificity = None
        pr_auc = None
        roc_auc = 0.0

        if is_binary:
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            try:
                roc_auc = float(roc_auc_score(y_test, probs[:, 1]))
                pr_auc = float(average_precision_score(y_test, probs[:, 1]))
            except Exception:
                roc_auc = 0.0
        else:
            try:
                roc_auc = float(roc_auc_score(y_test, probs, multi_class='ovr', average='weighted'))
            except Exception:
                roc_auc = 0.0

        # Extract feature importances / coefficients
        transformed_features = list(pipeline.named_steps['preprocessor'].get_feature_names_out())
        clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in transformed_features]

        fi_dict = {}
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            fi_dict = {feat: float(round(imp, 4)) for feat, imp in zip(clean_feature_names, importances)}
        elif hasattr(clf, 'coef_'):
            coefs = np.mean(np.abs(clf.coef_), axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
            norm_coefs = coefs / (np.sum(coefs) + 1e-9)
            fi_dict = {feat: float(round(c, 4)) for feat, c in zip(clean_feature_names, norm_coefs)}

        feature_importances[name] = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)[:8]

        results[name] = {
            'accuracy': round(acc, 4),
            'balanced_accuracy': round(bal_acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'specificity': round(specificity, 4) if specificity is not None else None,
            'f1_score': round(f1, 4),
            'roc_auc': round(roc_auc, 4),
            'pr_auc': round(pr_auc, 4) if pr_auc is not None else None,
            'cv_f1_mean': round(mean_cv_f1, 4),
            'confusion_matrix': cm
        }

        # Save pipeline to disk
        model_path = f"{model_dir}/{name}.joblib"
        joblib.dump(pipeline, model_path)
        saved_models[name] = model_path

        print(f"[{name.upper()}] Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | 5-Fold CV F1: {mean_cv_f1:.4f}")

        # Best model selection by test F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # 4. Save Complete Metadata & Evaluation Report
    metadata = {
        'module_id': module_id,
        'primary_model': best_model_name,
        'target_mapping': target_mapping,
        'features': features,
        'classes': classes,
        'evaluation': results,
        'saved_models': saved_models,
        'feature_importance': feature_importances,
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': {str(k): int(v) for k, v in pd.Series(y_encoded).value_counts().items()}
        }
    }

    meta_path = f"{model_dir}/metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Selected Best Model: {best_model_name.upper()} (F1: {best_f1:.4f})")
    print(f"Artifacts and evaluation saved to {meta_path}")
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate clinical health screening models.")
    parser.add_argument('--module', type=str, help='Specific module to train (anemia, blood_pressure, diabetes, obesity)')
    parser.add_argument('--all', action='store_true', help='Train all modules')
    args = parser.parse_args()

    if args.all:
        for module in DataLoader.DATA_CONFIG.keys():
            train_module(module)
    elif args.module:
        train_module(args.module)
    else:
        print("Please specify --module <module_id> or --all")
