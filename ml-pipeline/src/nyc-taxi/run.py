from pathlib import Path
from typing import Dict

from kedro.framework.context import KedroContext, load_package_context
from kedro.pipeline import Pipeline

from .pipeline import create_pipelines

from kedro_mlflow.framework.hooks import MlflowNodeHook, MlflowPipelineHook

class ProjectContext(KedroContext):

    project_name = "nyc-taxi"
    # `project_version` is the version of kedro used to generate the project
    project_version = "0.16.4"
    package_name = "nyc-taxi"
    hooks = (
        MlflowNodeHook(flatten_dict_params=False),
        MlflowPipelineHook(
            model_name="nyc-taxi", conda_env="/home/dwarszawski/Workspace/personal/dssconf2020/dssconf2020/ml-pipeline/src/requirements.txt",
        ),
    )

    def _get_pipelines(self) -> Dict[str, Pipeline]:
        return create_pipelines()


def run_package():
    # Entry point for running a Kedro project packaged with `kedro package`
    # using `python -m <project_package>.run` command.
    project_context = load_package_context(
        project_path=Path.cwd(), package_name=Path(__file__).resolve().parent.name
    )
    project_context.run()


if __name__ == "__main__":
    run_package()
