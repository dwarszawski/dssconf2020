from .pipelines.data_science import pipeline as ds
from .pipelines.data_engineering import pipeline as de


def create_pipelines(**kwargs):
    data_engineering_pipeline = de.create_pipeline()#.decorate(log_running_time)
    data_science_pipeline = ds.create_pipeline()#.decorate(log_running_time)

    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_engineering_pipeline + data_science_pipeline,
    }