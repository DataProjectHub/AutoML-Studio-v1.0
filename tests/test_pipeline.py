from pathlib import Path
from fastapi.testclient import TestClient
from main import run_pipeline
from automl_core.data_loader import read_config, load_data, split_data
from streamlit_app.app import app

def test_data_split():
    config = read_config()
    train, test, ytrain, ytest = split_data(load_data(config), config)
    assert len(train) == len(ytrain) and len(test) == len(ytest)
    assert len(train) > len(test) > 0

def test_end_to_end(tmp_path):
    config = tmp_path / "smoke.yaml"
    config.write_text("data_path: data/raw/synthetic_health_data.csv\ntarget_column: Health_Score\ntest_size: 0.2\nrandom_state: 42\nmodel_selection:\n  models: [LinearRegression]\n  cv_folds: 3\n")
    metrics = run_pipeline(config)
    assert metrics["best_model"] == "LinearRegression"
    assert all(key in metrics for key in ("RMSE", "MAE", "R2_Score"))
    client = TestClient(app)
    assert client.get("/health").json()["model_ready"] is True
    assert client.get("/").status_code == 200
    response = client.post("/predict", data={"Age": 35, "BMI": 23, "Exercise_Frequency": 3, "Diet_Quality": 7, "Sleep_Hours": 8, "Smoking_Status": 0, "Alcohol_Consumption": 1})
    assert response.status_code == 200, response.text
