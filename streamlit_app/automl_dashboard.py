
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import joblib
import io

st.set_page_config(page_title="AutoML Studio", layout="wide")

st.title("AutoML Studio")
st.subheader("Automated Regression & Model Evaluation Platform")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        target = st.selectbox(
            "Select the target column to predict",
            numeric_columns
        )

        st.info(f"Selected target: {target}")
        
        st.subheader("Automated Data Preprocessing")

        if st.button("Preprocess Dataset"):
            # Separate features and target
            X = df.drop(columns=[target])
            y = df[target]

            # Remove rows with missing target values
            valid_rows = y.notna()
            X = X.loc[valid_rows]
            y = y.loc[valid_rows]

            if len(X) < 5:
                st.error("Dataset is too small for preprocessing.")
            else:
                # Identify numerical and categorical columns
                numeric_features = X.select_dtypes(
                    include="number"
                ).columns.tolist()

                categorical_features = X.select_dtypes(
                    exclude="number"
                ).columns.tolist()

                # Split before fitting preprocessing
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Numerical preprocessing
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                # Categorical preprocessing
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore"
                    ))
                ])

                preprocessor = ColumnTransformer([
                    ("numeric", numeric_pipeline, numeric_features),
                    ("categorical", categorical_pipeline, categorical_features)
                ])

                # Learn transformations from training data only
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)

                st.success("Preprocessing completed successfully!")

                st.write("Training rows:", len(X_train))
                st.write("Testing rows:", len(X_test))
                st.write("Numerical features:", numeric_features)
                st.write("Categorical features:", categorical_features)
                
                st.subheader("Automated Model Training")

                models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest": RandomForestRegressor(
                        n_estimators=100, random_state=42
                    ),
                    "Gradient Boosting": GradientBoostingRegressor(
                        random_state=42
                    )
                }

                results = []
                best_score = float("inf")
                best_model_name = None

                for name, model in models.items():

                    # Preprocessing is fitted separately within each fold
                    pipeline = Pipeline([
                        ("preprocessor", preprocessor),
                        ("model", model)
                    ])

                    scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=5,
                        scoring="neg_root_mean_squared_error"
                    )

                    mean_rmse = -scores.mean()

                    results.append({
                        "Model": name,
                        "CV RMSE": round(mean_rmse, 4)
                    })

                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_model_name = name

                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values("CV RMSE")

                st.subheader("Model Comparison")
                st.dataframe(results_df, hide_index=True)

                st.success(
                    f"Best Model: {best_model_name}"
                )
                
                # Train the winning model on the full training set
                best_pipeline = Pipeline([
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(models[best_model_name]))
                ])

                best_pipeline.fit(X_train, y_train)

                # Predict on unseen test data
                y_pred = best_pipeline.predict(X_test)

                # Calculate evaluation metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.subheader("Final Model Evaluation")

                col1, col2, col3 = st.columns(3)

                col1.metric("R² Score", f"{r2:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")

                # Compare actual and predicted values
                comparison_df = pd.DataFrame({
                    "Actual": y_test.to_numpy(),
                    "Predicted": y_pred
                })

                st.subheader("Actual vs Predicted")
                st.dataframe(
                    comparison_df.head(20),
                    hide_index=True
                )

                # Download the entire trained pipeline
                model_buffer = io.BytesIO()
                joblib.dump(best_pipeline, model_buffer)

                st.download_button(
                    label="Download Best Model",
                    data=model_buffer.getvalue(),
                    file_name="automl_best_model.pkl",
                    mime="application/octet-stream"
                )



    else:
        st.warning("No numeric target columns found.")
