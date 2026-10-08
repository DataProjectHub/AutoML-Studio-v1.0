from pathlib import Path
import joblib
ROOT = Path(__file__).resolve().parents[1]
def load_artifacts(model_path=None):
    return joblib.load(model_path or ROOT / "models/best_model.pkl")
def predict_new_data(new_data_df):
    return load_artifacts().predict(new_data_df)
