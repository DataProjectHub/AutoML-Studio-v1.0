from pathlib import Path
from automl_core.data_loader import read_config, load_data, split_data
from automl_core.train import train_models, save_model
from automl_core.evaluate import evaluate_model
ROOT = Path(__file__).resolve().parent

def run_pipeline(config_path=None):
    config = read_config(config_path)
    X_train, X_test, y_train, y_test = split_data(load_data(config), config)
    results = train_models(X_train, y_train, config)
    winner = min(results, key=lambda name: results[name]["cv_rmse"])
    selected = results[winner]
    save_model(selected["model"], ROOT / "models/best_model.pkl")
    metrics = evaluate_model(selected["model"], X_test, y_test, ROOT / "reports/metrics.json", {"best_model": winner, "cv_rmse": selected["cv_rmse"], "model_comparison": {name: result["cv_rmse"] for name, result in results.items()}})
    print("Best model:", winner, "| Test metrics:", metrics)
    return metrics

if __name__ == "__main__": run_pipeline()
