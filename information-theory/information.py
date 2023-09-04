import multiprocessing

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
                    f, s = ordpy.fisher_shannon(window_df[feature], dx=4)
                    row[f'{feature}_entropy'] = h
                    row[f'{feature}_complexity'] = c
                    row[f'{feature}_fisher'] = f
                    row[f'{feature}_shannon'] = s
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


class InformationHandleFile():
    def __init__(self, path, window=30, shift=1, dx=3, dy=1, taux=1, tauy=1):
        # super().__init__()
        self.__window = window
        self.__shift = shift
        self.__dx = dx
        self.__dy = dy
        self.__taux = taux
        self.__tauy = tauy
        # self.__df = df
        # self.__fileout = fileout
        self.__path = path

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
        return self.__dx, self.__dy, self.__taux, self.__tauy

    @staticmethod
    def __run(df, fileout, window):
        sliding_window_df = InformationHandleFile.get_sub_lists(df, window)
        new_df = None
        new_df_sz = 0
        for window_df in sliding_window_df:
            row = {}
            for feature in window_df.columns:
                h, c = ordpy.complexity_entropy(window_df[feature])
                f, s = ordpy.fisher_shannon(window_df[feature])
                row[feature] = np.mean(window_df[feature].values)
                row[f'{feature}_entropy'] = h
                row[f'{feature}_complexity'] = c
                row[f'{feature}_fisher'] = f
                row[f'{feature}_shannon'] = s
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                new_df.loc[new_df_sz] = row
            new_df_sz += 1
        new_df.to_csv(fileout, index=False)

    def process_file(self, driver, num_files):
        threads = []
        for i in range(1, num_files+1, 1):
            filein = f'{self.__path}/{driver}/All_{i}.csv'
            fileout = f'{self.__path}/{driver}/All_{i}_inf.csv'
            df = pd.read_csv(filein)
            p = multiprocessing.Process(target=self.__run,
                                        args=(df, fileout, config.inf_window))
            p.start()
            threads.append(p)
        return threads

    def create_inf_measures_dataset(self):
        thread_pool = self.process_file('A', 8)
        thread_pool += self.process_file('B', 8)
        thread_pool += self.process_file('C', 5)
        thread_pool += self.process_file('D', 9)

        len_pool = len(thread_pool)
        print(len_pool)

        for i, thread in enumerate(thread_pool, start=1):
            thread.join()
            print(f'{i}/{len_pool}')


if __name__ == '__main__':
    InformationHandleFile(path='../../ThisCarIsMineNormalized').create_inf_measures_dataset()
