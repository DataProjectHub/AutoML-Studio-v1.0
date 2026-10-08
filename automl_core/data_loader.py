from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[1]
def read_config(config_path=None):
    path = Path(config_path) if config_path else ROOT / "automl_core/config/config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
def load_data(config):
    path = Path(config["data_path"])
    if not path.is_absolute(): path = ROOT / path
    df = pd.read_csv(path)
    target = config["target_column"]
    if target not in df.columns: raise ValueError(f"Target column not found: {target}")
    return df
def split_data(df, config):
    X = df.drop(columns=[config["target_column"]])
    y = df[config["target_column"]]
    return train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])
