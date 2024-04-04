import pandas as pd
import joblib
from scripts.config import (
    MODEL_PATH,
    CATEGORY_FEATURES,
    CONTINUOUS_FEATURES,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scripts.preprocess import preprocess, make_scaler, make_encoder
import mlflow


def split_data(
    df: pd.DataFrame,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train(
    df: pd.DataFrame,
    run_name: str,
    categorical_columns: list = CATEGORY_FEATURES,
    continuous_columns: list = CONTINUOUS_FEATURES,
) -> dict:
    experiment_name = "Churn Prediction"
    mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=f"run_{run_name}"
    ):
        mlflow.autolog()
        X_train, X_test, y_train, y_test = split_data(df)
        make_encoder(X_train, categorical_columns)
        make_scaler(X_train, continuous_columns)
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)
        model = make_model(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        metrics = {"testing_accuracy": accuracy, "testing_f1": f1, "mse": mse}
        params = model.get_params()
        mlflow.sklearn.log_model(model, "random forest regressor")
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mlflow.sklearn.save_model(
            model,
            "models/mlflow/models/random_forest_regressor",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    return metrics


def make_model(
    X_train: pd.DataFrame, y_train: pd.Series, path: str = MODEL_PATH
) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    return model
