from random import randint
import pandas as pd
import config


class GetData:
    def __init__(self, path_dataset, label_feature_name, driver_feature_name, features=config.features, trips=None):
        self.__path_dataset = path_dataset
        self.__label = label_feature_name
        self.__driver = driver_feature_name
        self.__features = features

        # path_dataset = '../../ThisCarIsMine'
        list_df_a = []
        list_df_b = []
        list_df_c = []
        list_df_d = []
        list_df_a_inf = []
        list_df_b_inf = []
        list_df_c_inf = []
        list_df_d_inf = []
        # read all files
        if trips is None:
            for i in range(1, 1 + 8):
                list_df_a.append(pd.read_csv(path_dataset + f'/A/All_{i}.csv')[features])
                list_df_a_inf.append(pd.read_csv(path_dataset + f'/A/All_{i}_inf.csv'))
            for i in range(1, 1 + 8):
                list_df_b.append(pd.read_csv(path_dataset + f'/B/All_{i}.csv')[features])
                list_df_b_inf.append(pd.read_csv(path_dataset + f'/B/All_{i}_inf.csv'))
            for i in range(1, 1 + 5):
                list_df_c.append(pd.read_csv(path_dataset + f'/C/All_{i}.csv')[features])
                list_df_c_inf.append(pd.read_csv(path_dataset + f'/C/All_{i}_inf.csv'))
            for i in range(1, 1 + 9):
                list_df_d.append(pd.read_csv(path_dataset + f'/D/All_{i}.csv')[features])
                list_df_d_inf.append(pd.read_csv(path_dataset + f'/D/All_{i}_inf.csv'))
        else:
            for i in trips:
                list_df_a.append(pd.read_csv(path_dataset + f'/A/All_{i}.csv')[features])
                list_df_b.append(pd.read_csv(path_dataset + f'/B/All_{i}.csv')[features])
                list_df_c.append(pd.read_csv(path_dataset + f'/C/All_{i}.csv')[features])
                list_df_d.append(pd.read_csv(path_dataset + f'/D/All_{i}.csv')[features])
                list_df_a_inf.append(pd.read_csv(path_dataset + f'/A/All_{i}_inf.csv'))
                list_df_b_inf.append(pd.read_csv(path_dataset + f'/B/All_{i}_inf.csv'))
                list_df_c_inf.append(pd.read_csv(path_dataset + f'/C/All_{i}_inf.csv'))
                list_df_d_inf.append(pd.read_csv(path_dataset + f'/D/All_{i}_inf.csv'))
        self.__DFA = pd.concat(list_df_a)
        self.__DFB = pd.concat(list_df_b)
        self.__DFC = pd.concat(list_df_c)
        self.__DFD = pd.concat(list_df_d)
        self.__DFA_inf = pd.concat(list_df_a_inf)
        self.__DFB_inf = pd.concat(list_df_b_inf)
        self.__DFC_inf = pd.concat(list_df_c_inf)
        self.__DFD_inf = pd.concat(list_df_d_inf)

    def get_sample(self, sample_size=50, driver_target='A'):
        data_a = self.__DFA.copy()
        data_b = self.__DFB.copy()
        data_c = self.__DFC.copy()
        data_d = self.__DFD.copy()
        if sample_size > 0:
            i = randint(0, max(data_a.shape[0] - sample_size, 0))
            data_a = data_a[i:i + sample_size]
            i = randint(0, max(data_b.shape[0] - sample_size, 0))
            data_b = data_b[i:i + sample_size]
            i = randint(0, max(data_c.shape[0] - sample_size, 0))
            data_c = data_c[i:i + sample_size]
            i = randint(0, max(data_d.shape[0] - sample_size, 0))
            data_d = data_d[i:i + sample_size]
        data_a[self.__label] = [int(driver_target == 'A')] * data_a.shape[0]
        data_b[self.__label] = [int(driver_target == 'B')] * data_b.shape[0]
        data_c[self.__label] = [int(driver_target == 'C')] * data_c.shape[0]
        data_d[self.__label] = [int(driver_target == 'D')] * data_d.shape[0]
        data_a[self.__driver] = ['A'] * data_a.shape[0]
        data_b[self.__driver] = ['B'] * data_b.shape[0]
        data_c[self.__driver] = ['C'] * data_c.shape[0]
        data_d[self.__driver] = ['D'] * data_d.shape[0]
        return pd.concat([data_a, data_b, data_c, data_d])

    def get_sample_inf(self, sample_size=50, driver_target='A'):
        data_a = self.__DFA_inf.copy()
        data_b = self.__DFB_inf.copy()
        data_c = self.__DFC_inf.copy()
        data_d = self.__DFD_inf.copy()
        if sample_size > 0:
            i = randint(0, max(data_a.shape[0] - sample_size, 0))
            data_a = data_a[i:i + sample_size]
            i = randint(0, max(data_b.shape[0] - sample_size, 0))
            data_b = data_b[i:i + sample_size]
            i = randint(0, max(data_c.shape[0] - sample_size, 0))
            data_c = data_c[i:i + sample_size]
            i = randint(0, max(data_d.shape[0] - sample_size, 0))
            data_d = data_d[i:i + sample_size]
        data_a[self.__label] = [int(driver_target == 'A')] * data_a.shape[0]
        data_b[self.__label] = [int(driver_target == 'B')] * data_b.shape[0]
        data_c[self.__label] = [int(driver_target == 'C')] * data_c.shape[0]
        data_d[self.__label] = [int(driver_target == 'D')] * data_d.shape[0]
        data_a[self.__driver] = ['A'] * data_a.shape[0]
        data_b[self.__driver] = ['B'] * data_b.shape[0]
        data_c[self.__driver] = ['C'] * data_c.shape[0]
        data_d[self.__driver] = ['D'] * data_d.shape[0]
        return pd.concat([data_a, data_b, data_c, data_d])

    def get_all(self, driver_target='A'):
        return self.get_sample(sample_size=-1, driver_target=driver_target)

    def get_all_inf(self, driver_target='A'):
        return self.get_sample_inf(sample_size=-1, driver_target=driver_target)
