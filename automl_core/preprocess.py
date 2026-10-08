from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X):
    numerical = X.select_dtypes(include="number").columns.tolist()
    categorical = X.columns.difference(numerical).tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
def preprocess_data(X_train, X_test):
    transformer = build_preprocessor(X_train)
    return transformer.fit_transform(X_train), transformer.transform(X_test), transformer
