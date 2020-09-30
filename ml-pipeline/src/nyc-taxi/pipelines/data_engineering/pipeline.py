from kedro.pipeline import Pipeline, node

from .nodes import preprocess

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess,
                inputs="trips",
                outputs="preprocessed_trips",
                name="feature_engineering",
            ),
        ]
    )