from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import json
import numpy as np

def evaluate_model(model, X_test, y_test, report_path="reports/metrics.json", metadata=None):
    predictions = model.predict(X_test)
    metrics = {"RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))), "MAE": float(mean_absolute_error(y_test, predictions)), "R2_Score": float(r2_score(y_test, predictions))}
    if metadata: metrics.update(metadata)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
