from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, split_data, train_model, train_explainer

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_trips", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test", "feature_names"],
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor"),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
            ),
            node(
                func=train_explainer,
                inputs=["regressor", "feature_names", "X_train", "X_test", "y_test"],
                outputs="explainer"),
        ]
    )