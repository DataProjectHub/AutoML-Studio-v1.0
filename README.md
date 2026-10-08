# AutoML Studio v1.0
### Automated Regression & Model Evaluation Platform

AutoML Studio is an independent machine learning project designed to simplify regression model development through an interactive Streamlit dashboard.

Users can upload CSV datasets, select a numerical target variable, automatically preprocess features, compare regression algorithms, evaluate model performance, and download the best trained pipeline.

## Key Features

- **CSV Upload:** Upload and preview tabular datasets.
- **Automated Preprocessing:** Missing-value imputation, numerical scaling, and categorical encoding.
- **Model Training:** Linear Regression, Random Forest, and Gradient Boosting.
- **Cross-Validation:** Five-fold cross-validation for model comparison.
- **Best Model Selection:** Automatically selects the model with the lowest cross-validation RMSE.
- **Model Evaluation:** R², RMSE, MAE, and actual-versus-predicted results.
- **Model Export:** Download the trained preprocessing and prediction pipeline as a `.pkl` file.

## Technology Stack

Python | Streamlit | Pandas | NumPy | Scikit-learn | Joblib | FastAPI

## Machine Learning Workflow

CSV Dataset → Target Selection → Train/Test Split → Automated Preprocessing → Cross-Validation → Best Model Selection → Test Evaluation → Model Export

Preprocessing is fitted within each cross-validation training fold to reduce data leakage.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/DataProjectHub/AutoML-Studio-v1.0.git
cd AutoML-Studio-v1.0
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch AutoML Studio:

```bash
python -m streamlit run streamlit_app/automl_dashboard.py
```

Open `http://localhost:8501` in your browser.

## Tested Use Cases

**Health Score Regression**

- Dataset: 1,000 records
- Best model: Linear Regression
- Test R²: 0.8090
- Test RMSE: 6.1026
- Test MAE: 4.6528

These results are from local testing on a demonstration dataset, not clinical validation.

**Bike Sharing Demand**

CSV upload and dataset preview tested with 17,379 records. Full model evaluation remains to be validated, including leakage prevention and time-aware splitting.

## Additional Components

The repository also contains a configurable Python regression pipeline and a FastAPI-based Health Score Predictor demonstration.

These components are separate from the main Streamlit dashboard.

## Current Limitations

- Supports regression, not classification.
- Requires a numerical target variable.
- Uses a fixed set of three algorithms in the Streamlit dashboard.
- Uses a random train/test split, which is not suitable for every dataset.
- Does not automatically detect identifiers, target leakage, or time-series structure.
- Downloaded models must be loaded only from trusted sources.

## Roadmap — Future Enhancements

- Dataset validation and feature exclusion
- Batch prediction and downloadable prediction reports
- Classification support
- Hyperparameter optimization
- Explainable AI with SHAP
- Time-series-aware validation
- Model tracking and versioning
- Improved experiment management

## Project Status

**Version 1.0 — Initial regression dashboard release candidate**

The core Streamlit workflow has been tested locally. Clean-environment installation and cloud deployment require final verification.

## Author
DataProjectHub

Independent machine learning and AI development project by Pooja Anilkumar
https://www.linkedin.com/in/pooja-a-8b678637/
