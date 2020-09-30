
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from alibi.explainers import AnchorTabular
import dill

import mlflow
import mlflow.sklearn


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    """Splits data into training and test sets.
        Args:
            data: Source data.
            parameters: Parameters defined in parameters.yml.
        Returns:
            A list containing split data.
    """
    X = data[[
        'pickup_hour',
        'sin_pickup_hour',
        'cos_pickup_hour',
        'night_hours',
        'weekday',
        'weekend',
        'passenger_count'
    ]]

    feature_names = X.columns.values

    X = X.values
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    return [X_train, X_test, y_train, y_test, feature_names]

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train the linear regression model.
        Args:
            X_train: Training data of independent features.
            y_train: Training data for price.
        Returns:
            Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)



    artifact_path = mlflow.get_artifact_uri()
    mlflow.sklearn.log_model(regressor, "model")
    mlflow.log_artifact("src/nyc-taxi/pipelines/data_science/requirements.txt", f"model")
    print(mlflow.active_run().info.run_id)

    return regressor


def evaluate_model(regressor: LinearRegression, X_test: np.ndarray, y_test: np.ndarray):
    """Calculate the coefficient of determination and log the result.
        Args:
            regressor: Trained model.
            X_test: Testing data of independent features.
            y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)

    # mlflow.log_metric("accuracy score", accuracy_score(y_test, y_pred))
    # mlflow.log_metric("recall score", recall_score(y_test, y_pred))
    # mlflow.log_metric("precision score", precision_score(y_test, y_pred))
    # mlflow.log_metric("f1 score", f1_score(y_test, y_pred))
    mlflow.log_metric("r2 score", r2_score(y_test, y_pred))

def train_explainer(regressor: LinearRegression, feature_names: List[str], X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    predict_fn = lambda x: regressor.predict(x)

    explainer = AnchorTabular(predict_fn, feature_names)
    explainer.fit(X_train)

    file_path=""
    with open("explainer.dill", "wb") as file:
        dill.dump(explainer, file)
        file_path = file.name

    mlflow.log_artifact("explainer.dill", "model")

    print(np.where(y_test == 1)[0])
    #probe = np. array([40.316667556762695, 0.5605325219195545, 0.350, 0, 3, 1, 5], dtype=float)
    probe = np. array(X_test[700], dtype=float)
    explanation = explainer.explain(probe, threshold=0.2)

    print('Anchor: %s' % (' AND '.join(explanation['names'])))
    print('Precision: %.2f' % explanation['precision'])
    print('Coverage: %.2f' % explanation['coverage'])
    print(explanation)
    return explainer

# kedro install
# kedro run
# kedro viz