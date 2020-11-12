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
    df_neg = df_neg.head(1000)

    dataframe = pd.concat([df_pos, df_neg], ignore_index=True)

    return dataframe
