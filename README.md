# AI-Based Early Prediction of Health Diseases

Educational machine-learning framework designed for multi-disease early risk stratification and health assessment.

> **CLINICAL DISCLAIMER & INTENDED USE**:
> This software is an educational prototype and research demonstrator developed for academic purposes. It is **NOT** a clinically certified diagnostic device, medical software (SaMD), or a substitute for professional clinical judgment, laboratory testing, or physician consultation. The probabilistic scores represent statistical associations within historical training datasets and must never be interpreted as calibrated clinical diagnoses.

---

## 1. System Architecture & Features

The platform provides end-to-end multi-disease screening across four disease modules:
- **Blood Pressure / Hypertension Risk**: Stratification based on demographic and laboratory markers (Age, Sex, BMI, Hemoglobin, Genetic Pedigree Coefficient, Chronic Kidney Disease).
- **Type 2 Diabetes Risk**: High-imbalance risk detection using HbA1c, fasting Blood Glucose, BMI, and Age.
- **Anemia Risk**: Classification from Complete Blood Count (CBC) parameters (Hemoglobin, MCH, MCHC, MCV, Gender).
- **Obesity & Weight Category**: Multi-class classification (7 categories) from anthropometric and lifestyle indicators.

### Key Capabilities
- **Multi-Model Consensus**: Runs inference across **Logistic Regression**, **Decision Tree**, and **Random Forest** simultaneously.
- **Transparent Model Evaluation**: Live evaluation metrics dashboard with 5-fold Stratified Cross-Validation F1, Accuracy, Precision, Recall, Specificity, ROC-AUC, PR-AUC, and visual Confusion Matrices.
- **Patient-Level Explainability**: Local factor impact indicators showing statistical deviation from baseline cohorts with clinical guidance notes (clarifying correlation vs causation).
- **Robust Clinical Validation**: Strict input validation with physiological bounds, acceptable step increments, and unit labels.
- **Privacy-First History**: Client-side localStorage session history with instant deletion and privacy controls.

---

## 2. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11 & 3.12)
- Virtual environment (recommended)

```bash
# Clone the repository
git clone https://github.com/p-srishanth-reddy/ML_based_health_prediction_system.git
cd ML_based_health_prediction_system

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Training the Machine Learning Models

The training pipeline implements 5-fold Stratified Cross-Validation on an 80/20 train-test split with deduplication, median imputation, standard scaling, and one-hot encoding.

```bash
# Train all 4 disease modules
python3 -m src.train_models --all

# Or train individual modules
python3 -m src.train_models --module blood_pressure
python3 -m src.train_models --module diabetes
python3 -m src.train_models --module anemia
python3 -m src.train_models --module obesity
```

Trained `.joblib` pipelines and metadata with evaluation metrics are saved to `models/<module_id>/`.

---

## 4. Running the Web Application

```bash
# Start the Flask web application
python3 app.py
```

The application will be accessible at `http://localhost:3000` (or `http://127.0.0.1:3000`).

---

## 5. Running the Automated Test Suite

```bash
# Run all unit, route, schema, and ML pipeline tests
pytest tests/ -v
```

All 17 automated tests verify:
- Data deduplication and feature alignment.
- Categorical mappings and standardization.
- Pipeline execution and standard scaling.
- Multi-model consensus generation.
- Physiological range validation and boundary rejection.
- Route accessibility, JSON APIs, and privacy endpoints.

---

## 6. Directory Structure

```
├── app.py                      # Flask backend, validation schemas, API routes
├── requirements.txt            # Python dependencies
├── src/
│   ├── data_loader.py          # Data ingestion, deduplication, schema alignment
│   ├── preprocessing.py        # Sklearn Pipeline & ColumnTransformer definition
│   ├── train_models.py         # 5-fold CV training, metric computation & persistence
│   ├── predict.py              # Multi-model inference, consensus & factor explainability
│   └── module_registry.py      # Physiological limits, units, tooltips & clinical presets
├── data/                       # Raw CSV datasets
│   ├── anemia/anemia.csv
│   ├── blood_pressure/blood_press.csv
│   ├── diabetes/diabetes.csv
│   └── obesity/obesity.csv
├── models/                     # Serialized pipelines (.joblib) and evaluation metadata
│   ├── anemia/
│   ├── blood_pressure/
│   ├── diabetes/
│   └── obesity/
├── static/
│   ├── css/style.css           # Responsive clinical styling & UI enhancements
│   └── js/app.js               # Dynamic forms, consensus visuals, validation & history
├── templates/                  # Jinja2 templates
│   ├── base.html
│   ├── index.html
│   ├── prediction.html
│   ├── evaluation.html
│   └── about.html
└── tests/
    ├── test_routes.py          # Route, schema, and API integration tests
    └── test_ml_pipeline.py     # Pipeline, deduplication, and inference tests
```
