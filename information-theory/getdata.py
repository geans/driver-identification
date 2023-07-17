from random import randint
import pandas as pd
import config


class GetData:
    def __init__(self, path_dataset, label_feature_name, driver_feature_name):
        self.__label = label_feature_name
        self.__driver = driver_feature_name

        # path_dataset = '../../ThisCarIsMine'
        list_df_a = []
        list_df_b = []
        list_df_c = []
        list_df_d = []
        # read all files
        for i in range(1, 1 + 8):
            list_df_a.append(pd.read_csv(path_dataset + f'/A/All_{i}.csv')[config.features])
        for i in range(1, 1 + 8):
            list_df_b.append(pd.read_csv(path_dataset + f'/B/All_{i}.csv')[config.features])
        for i in range(1, 1 + 5):
            list_df_c.append(pd.read_csv(path_dataset + f'/C/All_{i}.csv')[config.features])
        for i in range(1, 1 + 9):
            list_df_d.append(pd.read_csv(path_dataset + f'/D/All_{i}.csv')[config.features])
        self.__DFA = pd.concat(list_df_a)
        self.__DFB = pd.concat(list_df_b)
        self.__DFC = pd.concat(list_df_c)
        self.__DFD = pd.concat(list_df_d)

    def get_sample(self, samplesize=50, driver_target='A'):
        data_a = self.__DFA.copy()
        data_b = self.__DFB.copy()
        data_c = self.__DFC.copy()
        data_d = self.__DFD.copy()
        if samplesize > 0:
            i = randint(0, max(data_a.shape[0] - samplesize, 0))
            data_a = data_a[i:i + samplesize]
            i = randint(0, max(data_b.shape[0] - samplesize, 0))
            data_b = data_b[i:i + samplesize]
            i = randint(0, max(data_c.shape[0] - samplesize, 0))
            data_c = data_c[i:i + samplesize]
            i = randint(0, max(data_d.shape[0] - samplesize, 0))
            data_d = data_d[i:i + samplesize]
        data_a[self.__label] = [int(driver_target == 'A')] * data_a.shape[0]
        data_b[self.__label] = [int(driver_target == 'B')] * data_b.shape[0]
        data_c[self.__label] = [int(driver_target == 'C')] * data_c.shape[0]
        data_d[self.__label] = [int(driver_target == 'D')] * data_d.shape[0]
        data_a[self.__driver] = ['A'] * data_a.shape[0]
        data_b[self.__driver] = ['B'] * data_b.shape[0]
        data_c[self.__driver] = ['C'] * data_c.shape[0]
        data_d[self.__driver] = ['D'] * data_d.shape[0]
        return pd.concat([data_a, data_b, data_c, data_d])
