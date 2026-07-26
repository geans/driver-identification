import pandas as pd


def split_data_to_window(series, window_size, shift):
    pivot = 0
    sub_series_list = []
    len_list = len(series)
    while pivot + window_size <= len_list:
        sub_series_list.append(series[pivot:pivot + window_size])
        pivot += shift
    return sub_series_list


def preprocessing_to_hc(series):
    series = series.loc[series.shift() != series]
    return series


def get_data_list(path):
    data_arr = [
        (pd.read_csv(path + '/A/All_1.csv'), 'A'),
        (pd.read_csv(path + '/A/All_2.csv'), 'A'),
        (pd.read_csv(path + '/A/All_3.csv'), 'A'),
        (pd.read_csv(path + '/A/All_4.csv'), 'A'),
        (pd.read_csv(path + '/A/All_5.csv'), 'A'),
        (pd.read_csv(path + '/A/All_6.csv'), 'A'),
        (pd.read_csv(path + '/A/All_7.csv'), 'A'),
        # (pd.read_csv(path + '/A/All_8.csv'), 'A'), # outlier
        #
        (pd.read_csv(path + '/B/All_1.csv'), 'B'),
        (pd.read_csv(path + '/B/All_2.csv'), 'B'),
        (pd.read_csv(path + '/B/All_3.csv'), 'B'),
        (pd.read_csv(path + '/B/All_4.csv'), 'B'),
        # (pd.read_csv(path + '/B/All_5.csv'), 'B'), # outlier
        (pd.read_csv(path + '/B/All_6.csv'), 'B'),
        (pd.read_csv(path + '/B/All_7.csv'), 'B'),
        (pd.read_csv(path + '/B/All_8.csv'), 'B'),
        #
        (pd.read_csv(path + '/C/All_1.csv'), 'C'),
        (pd.read_csv(path + '/C/All_2.csv'), 'C'),
        (pd.read_csv(path + '/C/All_3.csv'), 'C'),
        (pd.read_csv(path + '/C/All_4.csv'), 'C'),
        (pd.read_csv(path + '/C/All_5.csv'), 'C'),
        #
        (pd.read_csv(path + '/D/All_1.csv'), 'D'),
        (pd.read_csv(path + '/D/All_2.csv'), 'D'),
        (pd.read_csv(path + '/D/All_3.csv'), 'D'),
        (pd.read_csv(path + '/D/All_4.csv'), 'D'),
        (pd.read_csv(path + '/D/All_5.csv'), 'D'),
        (pd.read_csv(path + '/D/All_6.csv'), 'D'),
        (pd.read_csv(path + '/D/All_7.csv'), 'D'),
        (pd.read_csv(path + '/D/All_8.csv'), 'D'),
        (pd.read_csv(path + '/D/All_9.csv'), 'D'),
    ]
    return data_arr