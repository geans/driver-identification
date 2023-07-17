from sklearn import preprocessing
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


class LiteraturePreprocessing:
    def __init__(self, df, target='class', driver='driver'):
        self.__df = df
        self.__target = target
        self.__driver = driver

    def get_df(self):
        return self.__df

    def get_df_list(self):
        df = self.__df
        return [df[df[self.__driver] == 'A'], df[df[self.__driver] == 'B'],
                df[df[self.__driver] == 'C'], df[df[self.__driver] == 'D']]

    def remove_correlation(self, correlate_threshold=0.95):
        df = self.__df
        included = [self.__target, self.__driver]
        columns = list(df.drop([self.__target, self.__driver], axis=1).columns)
        for i in range(len(columns)):
            c1 = df[columns[i]]
            must_add = True
            for j in range(i + 1, len(columns), 1):
                c2 = df[columns[j]]
                if c1.corr(c2) > correlate_threshold:
                    must_add = False
                    break
            if must_add:
                included.append(columns[i])
        self.__df = df[included]

    def remove_miss_value(self):
        self.__df = self.__df.dropna(axis=0)

    def normalization(self):
        df = self.__df
        x = df.drop([self.__target, self.__driver], axis=1).values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        new_df = pd.DataFrame(x_scaled, columns=df.drop([self.__target, self.__driver], axis=1).columns)
        new_df[self.__target] = list(df[self.__target])
        new_df[self.__driver] = list(df[self.__driver])
        self.__df = new_df

    def remove_invariants(self):
        df = self.__df
        new_df = df.drop([self.__target, self.__driver], axis=1)
        new_df = new_df.loc[:, (new_df != new_df.iloc[0]).any()]
        new_df[self.__target] = list(df[self.__target])
        new_df[self.__driver] = list(df[self.__driver])
        self.__df = new_df

    def window(self, size=30):
        df = self.__df
        new_df = df.rolling(size, center=True, min_periods=1).mean()
        new_df[self.__target] = df[self.__target]
        new_df[self.__driver] = df[self.__driver]
        self.__df = new_df


class InformationPreprocessing:
    def __init__(self, df, target='class', driver='driver'):
        self.__df = df
        self.__target = target
        self.__driver = driver

    def get_df(self):
        return self.__df

    def get_df_list(self):
        df = self.__df
        return [df[df[self.__driver] == 'A'], df[df[self.__driver] == 'B'],
                df[df[self.__driver] == 'C'], df[df[self.__driver] == 'D']]

    def drop_consecutive_duplicates(self):
        df = self.__df
        self.__df = df.loc[(df.shift(-1) != df).any(axis=1)]
