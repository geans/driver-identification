import ordpy
import pandas as pd


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

    def __get_sublists(self, original_list, delta):
        pivot = 0
        sublists = []
        len_list = len(original_list)
        shift = 1
        while pivot + delta <= len_list:
            sublists.append(original_list[pivot:pivot + delta])
            pivot += shift
        return sublists

    def get_parameters(self):
        return self.__dx, self.__dy, self.__taux, self.__tauy

    def get_information_binary_label(self, df_, both=False, hc_size=30):
        new_df_list = []
        for df, driver in zip([df_[df_[self.__driver] == 'A'],
                               df_[df_[self.__driver] == 'B'],
                               df_[df_[self.__driver] == 'C'],
                               df_[df_[self.__driver] == 'D']],
                              ['A', 'B', 'C', 'D']):
            sliding_window_df = self.__get_sublists(df, hc_size)
            new_df = None
            new_df_sz = 0
            for window_df in sliding_window_df:
                row = {}
                for feature in window_df.drop([self.__label, self.__driver], axis=1).columns:
                    h, c = ordpy.complexity_entropy(window_df[feature], dx=self.__dx, dy=self.__dy, taux=self.__taux,
                                                    tauy=self.__tauy)
                    # f, s = ordpy.fisher_shannon(window_df[feature], dx=4)
                    row[f'{feature}_entropy'] = h
                    row[f'{feature}_complexity'] = c
                    # row[f'{feature}_fisher'] = f
                    # row[f'{feature}_shannon'] = s
                    if both:
                        if window_df[feature].dtypes == int:
                            row[feature] = window_df[feature].values[0]
                        else:
                            row[feature] = window_df[feature].values[0]
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
            sliding_window_df = self.__get_sublists(df, hc_size)
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
                        if window_df[feature].dtypes == int:
                            row[feature] = window_df[feature].values[0]
                        else:
                            row[feature] = window_df[feature].values[0]
                row[self.__label] = driver
                row[self.__driver] = driver
                if new_df is None:
                    new_df = pd.DataFrame([row])
                else:
                    new_df.loc[new_df_sz] = row
                new_df_sz += 1
            new_df_list.append(new_df)
        return pd.concat(new_df_list)
