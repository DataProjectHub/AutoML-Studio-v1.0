import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from .preprocess import build_preprocessor

def train_models(X_train, y_train, config=None):
    config = config or {}
    spec = {
        "LinearRegression": (LinearRegression(), {"model__fit_intercept": [True, False]}),
        "RandomForest": (RandomForestRegressor(random_state=42), {"model__n_estimators": [100, 200], "model__max_depth": [None, 10], "model__min_samples_split": [2, 5]}),
        "XGBoost": (XGBRegressor(random_state=42, verbosity=0, n_jobs=1), {"model__n_estimators": [100, 200], "model__max_depth": [3, 5], "model__learning_rate": [0.01, 0.1]})
    }
    settings = config.get("model_selection", {})
    names = settings.get("models", list(spec))
    if not names: raise ValueError("Select at least one model")
    results = {}
    for name in names:
        if name not in spec: raise ValueError(f"Unknown model: {name}")
        estimator, params = spec[name]
        pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        grid = GridSearchCV(pipeline, params, cv=settings.get("cv_folds", 5), scoring="neg_root_mean_squared_error", n_jobs=1)
        grid.fit(X_train, y_train)
        results[name] = {"model": grid.best_estimator_, "cv_rmse": float(-grid.best_score_), "parameters": grid.best_params_}
        print(f"{name}: cross-validation RMSE {results[name]['cv_rmse']:.4f}")
    return results

def save_model(model, file_path="models/best_model.pkl"):
    from pathlib import Path
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
