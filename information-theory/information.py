import multiprocessing
import time

import numpy as np
import ordpy
import pandas as pd

import config


class InformationHandle:
    def __init__(self, label_feature_name, driver_feature_name,
                 dx=4, dy=1, taux=1, tauy=1):
        self.__label = label_feature_name
        self.__driver = driver_feature_name
        self.__dx = dx
        self.__dy = dy
        self.__taux = taux
        self.__tauy = tauy
        self.__df = []

    @staticmethod
    def __get_sub_lists(original_list, delta):
        pivot = 0
        sub_lists = []
        len_list = len(original_list)
        shift = 1
        while pivot + delta <= len_list:
            sub_lists.append(original_list[pivot:pivot + delta])
            pivot += shift
        return sub_lists

    def get_parameters(self):
        return self.__dx, self.__dy, self.__taux, self.__tauy

    def get_information_binary_label(self, df_, both=False, hc_size=30):
        new_df_list = []
        for df, driver in zip([df_[df_[self.__driver] == 'A'],
                               df_[df_[self.__driver] == 'B'],
                               df_[df_[self.__driver] == 'C'],
                               df_[df_[self.__driver] == 'D']],
                              ['A', 'B', 'C', 'D']):
            sliding_window_df = self.__get_sub_lists(df, hc_size)
            new_df = None
            new_df_sz = 0
            for window_df in sliding_window_df:
                row = {}
                for feature in window_df.drop([self.__label, self.__driver], axis=1).columns:
                    h, c = ordpy.complexity_entropy(window_df[feature], dx=self.__dx, dy=self.__dy, taux=self.__taux,
                                                    tauy=self.__tauy)
                    s, f = ordpy.fisher_shannon(window_df[feature], dx=4)
                    row[f'{feature}_entropy'] = h
                    row[f'{feature}_complexity'] = c
                    row[f'{feature}_shannon'] = s
                    row[f'{feature}_fisher'] = f
                    if both:
                        # row[feature] = window_df[feature].values[0]
                        row[feature] = np.mean(window_df[feature].values)
                row[self.__label] = window_df[self.__label].values[0]
                row[self.__driver] = driver
                if new_df is None:
                    new_df = pd.DataFrame([row])
                else:
                    new_df.loc[new_df_sz] = row
                new_df_sz += 1
            new_df_list.append(new_df)
        return pd.concat(new_df_list)

    def get_information_by_driver(self, df_, both=False, hc_size=30):
        new_df_list = []
        for df, driver in zip([df_[df_[self.__driver] == 'A'],
                               df_[df_[self.__driver] == 'B'],
                               df_[df_[self.__driver] == 'C'],
                               df_[df_[self.__driver] == 'D']],
                              ['A', 'B', 'C', 'D']):
            sliding_window_df = self.__get_sub_lists(df, hc_size)
            new_df = None
            new_df_sz = 0
            for window_df in sliding_window_df:
                row = {}
                for feature in window_df.drop([self.__label, self.__driver], axis=1).columns:
                    h, c = ordpy.complexity_entropy(window_df[feature],
                                                    dx=self.__dx, dy=self.__dy, taux=self.__taux, tauy=self.__tauy)
                    f, s = ordpy.fisher_shannon(window_df[feature], dx=4)
                    row[f'{feature}_entropy'] = h
                    row[f'{feature}_complexity'] = c
                    row[f'{feature}_fisher'] = f
                    row[f'{feature}_shannon'] = s
                    if both:
                        # row[feature] = window_df[feature].values[0]
                        row[feature] = np.mean(window_df[feature].values)
                row[self.__label] = driver
                row[self.__driver] = driver
                if new_df is None:
                    new_df = pd.DataFrame([row])
                else:
                    new_df.loc[new_df_sz] = row
                new_df_sz += 1
            new_df_list.append(new_df)
        return pd.concat(new_df_list)


class InformationHandleFile:
    def __init__(self, path, path_out,  window, shift=1, dx=6):
        self.__window = window
        self.__shift = shift
        self.__dx = dx
        self.__path = path
        self.__path_out = path_out
        self.__time_hc = []
        self.__time_fs = []

    @staticmethod
    def get_sub_lists(original_list, delta):
        pivot = 0
        sub_lists = []
        len_list = len(original_list)
        shift = 1
        while pivot + delta <= len_list:
            sub_lists.append(original_list[pivot:pivot + delta])
            pivot += shift
        return sub_lists

    def get_parameters(self):
        return self.__dx

    @staticmethod
    def __run(df, fileout, window_size, dx):
        sliding_window_df = InformationHandleFile.get_sub_lists(df, window_size)
        new_df = None
        new_df_sz = 0
        time_hc = []
        time_fs = []
        for window_df in sliding_window_df:
            row = {}
            for feature in window_df.columns:
                window_without_duplicate = window_df[feature].loc[window_df[feature].shift() != window_df[feature]]
                if len(window_without_duplicate) < dx:
                    h = c = f = s = 'NaN'
                else:
                    t0 = time.time()
                    h, c = ordpy.complexity_entropy(window_without_duplicate, dx=dx)
                    tf = time.time()
                    time_hc.append(tf - t0)
                    #
                    t0 = time.time()
                    f, s = ordpy.fisher_shannon(window_without_duplicate, dx=dx)
                    tf = time.time()
                    time_fs.append(tf - t0)
                # row[feature] = window_df[feature].values[-1]
        #         row[f'{feature}_entropy'] = h
        #         row[f'{feature}_complexity'] = c
        #         row[f'{feature}_fisher'] = f
        #         row[f'{feature}_shannon'] = s
        #     if new_df is None:
        #         new_df = pd.DataFrame([row])
        #     else:
        #         new_df.loc[new_df_sz] = row
        #     new_df_sz += 1
        # new_df.to_csv(fileout, index=False)
        time_dict = {
            'time_hc': time_hc,
            'time_fs': time_fs
        }
        time_df = pd.DataFrame.from_dict(time_dict)
        time_df.to_csv(fileout+'.time', index=False)

    def __process_file(self, driver, num_files):
        threads = []
        for i in range(1, num_files+1, 1):
            filein = f'{self.__path}/{driver}/All_{i}.csv'
            fileout = f'{self.__path_out}/{driver}/All_{i}.csv'
            df = pd.read_csv(filein)
            p = multiprocessing.Process(target=self.__run,
                                        args=(df, fileout, self.__window, self.__dx))
            p.start()
            threads.append(p)
        return threads

    def create_inf_measures_dataset(self):
        thread_pool = self.__process_file('A', 8)
        thread_pool += self.__process_file('B', 8)
        thread_pool += self.__process_file('C', 5)
        thread_pool += self.__process_file('D', 9)
        #
        len_pool = len(thread_pool)
        print(f'{len_pool} threads. Window={self.__window}. dx={self.__dx}')
        #
        for i, thread in enumerate(thread_pool, start=1):
            thread.join()
            print(f'{i}/{len_pool}')


if __name__ == '__main__':
    path_in = '../../ThisCarIsMine'
    #
    path_out = '../../ThisCarIsMineInf_window20_dx4'
    InformationHandleFile(path=path_in,
                          path_out=path_out,
                          window=20,
                          dx=4
                          ).create_inf_measures_dataset()
    #
    path_out = '../../ThisCarIsMineInf_window120_dx5'
    InformationHandleFile(path=path_in,
                          path_out=path_out,
                          window=120,
                          dx=5
                          ).create_inf_measures_dataset()
    #
    path_out = '../../ThisCarIsMineInf_window300_dx6'
    InformationHandleFile(path=path_in,
                          path_out=path_out,
                          window=300,
                          dx=6
                          ).create_inf_measures_dataset()
    #
    path_out = '../../ThisCarIsMineInf_window720_dx6'
    InformationHandleFile(path=path_in,
                          path_out=path_out,
                          window=720,
                          dx=6
                          ).create_inf_measures_dataset()
    #
    path_out = '../../ThisCarIsMineInf_window900_dx7'
    InformationHandleFile(path=path_in,
                          path_out=path_out,
                          window=900,
                          dx=7
                          ).create_inf_measures_dataset()
