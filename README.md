# AutoML Studio — Automated Regression & Model Evaluation Platform

AutoML Studio is an end-to-end, configurable regression benchmark with automated preprocessing, cross-validated model selection, holdout evaluation, a saved inference pipeline, and a FastAPI prediction form.

## Project scope
AutoML Studio is the platform name. The included **Health Score Predictor** is an example regression use case, not a separate general-purpose prediction interface.

## Features
- CSV ingestion and train/test split from YAML configuration
- Numeric imputation and scaling; categorical imputation and one-hot encoding
- Linear Regression, Random Forest, and XGBoost hyperparameter search
- **Leakage-safe cross-validation:** preprocessing is fitted inside each training fold
- Winner selected by lowest cross-validation RMSE; final test set used only for evaluation
- JSON report with RMSE, MAE, R², and model comparison
- FastAPI prediction form and `/health` endpoint

## Quick start
```bash
python -m venv .venv
# Activate .venv for your operating system
pip install -r requirements.txt
python main.py
uvicorn streamlit_app.app:app --reload
```
Open http://127.0.0.1:8000. Training may take several minutes depending on your computer.

## Run tests
```bash
pytest -q
```

## Configuration
Edit `automl_core/config/config.yaml` to change the CSV path, target, test split, model list, or cross-validation folds. The included dataset is synthetic health-score data used as a regression demonstration, **not** a clinically validated predictor.

## Project layout
- `automl_core/` – data loading, preprocessing, training, evaluation
- `data/raw/` – example CSV
- `main.py` – full training workflow
- `streamlit_app/app.py` – **FastAPI**, despite the legacy folder name
- `tests/` – smoke and integration tests
- `reports/metrics.json` – generated metrics
- `models/best_model.pkl` – generated fitted pipeline (ignored by Git)

## Deployment
`render.yaml` trains a model during build and serves FastAPI with Uvicorn. Render build limits and free-tier availability may change; validate on the target host. This example is a demo and has no authentication or production-grade monitoring.

## Roadmap
- Rename legacy `streamlit_app` directory to `web_app`
- Add model registry and experiment tracking
- Add downloadable batch prediction and data-quality validation
- Add CI and model drift monitoring

## Limitations
This is a configurable AutoML-style **model search**, not a general-purpose AutoML framework. It supports regression with three model families and assumes the provided features are suitable for prediction.
