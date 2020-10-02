import pandas as pd
import numpy as np
# from functools import wraps
# from typing import Callable
# import time
# import logging


# def log_running_time(func: Callable) -> Callable:
#     """Decorator for logging node execution time.
#
#         Args:
#             func: Function to be executed.
#         Returns:
#             Decorator for logging the running time.
#
#     """
#
#     @wraps(func)
#     def with_time(*args, **kwargs):
#         log = logging.getLogger(__name__)
#         t_start = time.time()
#         result = func(*args, **kwargs)
#         t_end = time.time()
#         elapsed = t_end - t_start
#         log.info("Running %r took %.2f seconds", func.__name__, elapsed)
#         return result
#
#     return with_time

def extract_hour(row):
    return float(int(row['pickup_date'].hour) + int(row['pickup_date'].minute)/60)

def night_hour(row):
    return row['pickup_hour'] < 4

def weekday(row):
    return row['pickup_date'].weekday()

def weekday(row):
    return row['pickup_date'].weekday()

def is_weekend(row):
    return row['weekday']>4

def is_airport_destination(row):
    return row['weekday']>4


#@log_running_time
def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['tpep_pickup_datetime'] = pd.to_datetime(dataframe['tpep_pickup_datetime'])
    dataframe.rename(columns={'tpep_pickup_datetime': 'pickup_date'}, inplace=True)
    dataframe['pickup_hour'] = dataframe.apply (lambda row: extract_hour(row), axis=1)

    dataframe['sin_pickup_hour'] = np.sin(dataframe['pickup_hour'])
    dataframe['cos_pickup_hour'] = np.cos(dataframe['pickup_hour'])
    dataframe['night_hours'] = dataframe.apply (lambda row: night_hour(row), axis=1)
    dataframe['weekday'] = dataframe.apply (lambda row: weekday(row), axis=1)
    dataframe['weekend'] = dataframe.apply (lambda row: is_weekend(row), axis=1)
    dataframe['passenger_count'] = dataframe['passenger_count'].astype(int)
    dataframe['label'] = dataframe.apply(lambda row: 1 if row['DOLocationID'] in [1, 132, 138] else 0, axis=1)
    dataframe = dataframe[[
        'pickup_hour',
        'sin_pickup_hour',
        'cos_pickup_hour',
        'night_hours',
        'weekday',
        'weekend',
        'passenger_count',
        'label'
    ]]

    df_pos = dataframe[dataframe.label.eq(1)]
    df_neg = dataframe[dataframe.label.eq(0)]
    df_neg = df_neg.head(500)

    dataframe = pd.concat([df_pos, df_neg], ignore_index=True)

    return dataframe