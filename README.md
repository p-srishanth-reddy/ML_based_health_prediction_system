# AI-Based Early Prediction of Health Diseases

## Setup
1. Install dependencies:
   `pip install -r requirements.txt`

2. Download datasets as per `data/README.md`.

3. Train models:
   `python -m src.train_models --all`

4. Run application:
   `python app.py`

## Project Structure
- `app.py`: Flask server
- `src/`: Core logic (preprocessing, training, prediction)
- `models/`: Saved model artifacts (.joblib)
- `templates/`: HTML frontend
- `static/`: CSS/JS assets
