#!/usr/bin/python3

import config
from plot_results import PlotResults
from datetime import timedelta
from getdata import GetData
from information import InformationHandle
from numpy import mean
from scipy.stats import sem
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import time

from data_preprocessing import LiteraturePreprocessing

import warnings

warnings.filterwarnings("ignore")


PATH_DATASET = config.path_dataset
SAMPLE_SIZE = config.sample_size
HC_SIZE = config.hc_size
NUM_REPETITIONS = config.num_repetitions
dx, dy, tx, ty = config.entropy_complexity_parameters
features = config.features
DEBUG_ON_SCREEN = config.debug_on_screen

# Create empty file
f = open(f'output__{SAMPLE_SIZE}.txt', 'w')
f.close()


def mdebug(*objects, sep=' ', end='\n', file=None, flush=False):
    if DEBUG_ON_SCREEN:
        print(*objects, sep=sep, end=end, file=file, flush=flush)
    output_file = open(f'output__{SAMPLE_SIZE}.txt', 'a')
    for obj in objects:
        output_file.write(str(obj))
        output_file.write(sep)
    output_file.write(end)
    output_file.close()


classifier_names = [
    "kNN",
    "Linear SVM",
    "RBF SVM",
    "D. Tree",
    "R. Forest",
    "MLP",
    "N. Bayes",
]

classifiers = [
    KNeighborsClassifier(10),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=.9, C=1),
    DecisionTreeClassifier(max_depth=51),
    RandomForestClassifier(max_depth=51, n_estimators=10, max_features=10),
    MLPClassifier(max_iter=1000, hidden_layer_sizes=(51,)),
    GaussianNB(),
]


# PRE-PROCESSING

# class Preprocessing:
#     def __init__(self, df, label_feature_name, driver_feature_name):
#         self.__df = df
#         self.__target = label_feature_name
#         self.__driver = driver_feature_name
#
#     def get_df(self):
#         return self.__df
#
#     def get_df_list(self):
#         df = self.__df
#         return [df[df[self.__driver] == 'A'], df[df[self.__driver] == 'B'],
#                 df[df[self.__driver] == 'C'], df[df[self.__driver] == 'D']]
#
#     def remove_correlation(self, correlate_threshold=0.95):
#         df = self.__df
#         correlate_threshold = 0.95
#         included = [self.__target, self.__driver]
#         columns = list(df.drop([self.__target, self.__driver], axis=1).columns)
#         for i in range(len(columns)):
#             c1 = df[columns[i]]
#             must_add = True
#             for j in range(i + 1, len(columns), 1):
#                 c2 = df[columns[j]]
#                 if c1.corr(c2) > 0.95:
#                     must_add = False
#                     break
#             if must_add:
#                 included.append(columns[i])
#         self.__df = df[included]
#
#     def remove_miss_value(self):
#         self.__df = self.__df.dropna(axis=0)
#
#     def normalization(self):
#         df = self.__df
#         x = df.drop([self.__target, self.__driver], axis=1).values  # returns a numpy array
#         min_max_scaler = preprocessing.MinMaxScaler()
#         x_scaled = min_max_scaler.fit_transform(x)
#         new_df = pd.DataFrame(x_scaled, columns=df.drop([self.__target, self.__driver], axis=1).columns)
#         new_df[self.__target] = df[self.__target]
#         new_df[self.__driver] = df[self.__driver]
#         self.__df = new_df
#
#     def remove_invariants(self):
#         df = self.__df
#         new_df = df.drop([self.__target, self.__driver], axis=1)
#         new_df = new_df.loc[:, (new_df != new_df.iloc[0]).any()]
#         new_df[self.__target] = df[self.__target]
#         new_df[self.__driver] = df[self.__driver]
#         self.__df = new_df
#
#     def window(self, size=30):
#         df = self.__df
#         new_df = df.rolling(size, center=True, min_periods=1).mean()
#         new_df[self.__target] = df[self.__target]
#         new_df[self.__driver] = df[self.__driver]
#         self.__df = new_df
#
#
# class InfPreprocessing:
#     def __init__(self, df, label_feature_name, driver_feature_name):
#         self.__df = df
#         self.__target = label_feature_name
#         self.__driver = driver_feature_name
#
#     def get_df(self):
#         return self.__df
#
#     def get_df_list(self):
#         df = self.__df
#         return [df[df[self.__driver] == 'A'], df[df[self.__driver] == 'B'],
#                 df[df[self.__driver] == 'C'], df[df[self.__driver] == 'D']]
#
#     def drop_consecutive_duplicates(self):
#         self.__remove_invariants()
#         self.__remove_correlation()
#         df = self.__df
#         self.__df = df.loc[(df.shift(-1) != df).any(axis=1)]
#         # for col in df.drop([self.__target, self.__driver], axis=1).columns:
#         #     df = df.loc[df[col].shift(-1) != df[col]]
#
#     def __remove_invariants(self):
#         df = self.__df
#         new_df = df.drop([self.__target, self.__driver], axis=1)
#         new_df = new_df.loc[:, (new_df != new_df.iloc[0]).any()]
#         new_df[self.__target] = df[self.__target]
#         new_df[self.__driver] = df[self.__driver]
#         self.__df = new_df
#
#     def __remove_correlation(self, correlate_threshold=0.95):
#         df = self.__df
#         correlate_threshold = 0.95
#         included = [self.__target, self.__driver]
#         columns = list(df.drop([self.__target, self.__driver], axis=1).columns)
#         for i in range(len(columns)):
#             c1 = df[columns[i]]
#             must_add = True
#             for j in range(i + 1, len(columns), 1):
#                 c2 = df[columns[j]]
#                 if c1.corr(c2) > 0.95:
#                     must_add = False
#                     break
#             if must_add:
#                 included.append(columns[i])
#         self.__df = df[included]


def classifiers_train_test(X, y, _score_dic, _time_dic, k_fold):
    def classifier_handle(name, clf, X, y):
        t0 = time.time()
        result_dict = cross_validate(clf, X, y, cv=k_fold, return_estimator=True, return_train_score=True)
        time_fit = (time.time() - t0) / k_fold
        score_dic[name] = score_dic.get(name, []) + [mean(result_dict['test_score'])]
        time_dic[name] = time_dic.get(name, []) + [time_fit]

    running_process = []
    manager = multiprocessing.Manager()
    score_dic = manager.dict(_score_dic)
    time_dic = manager.dict(_time_dic)
    for name, clf in zip(classifier_names, classifiers):
        p = multiprocessing.Process(target=classifier_handle,
                                    args=(name, clf, X, y))
        p.start()
        # p.join()
        running_process.append(p)
    for p in running_process:
        p.join()
    mdebug(f'''
    [partial data]
        score_dic = {score_dic}
        time_dic  = {time_dic}
    [end partial data]
    ''')
    return score_dic, time_dic


if __name__ == '__main__':
    program_time = time.time()
    time_literature = {}
    score_literature = {}
    time_dataprocessing = []

    time_inf_teory = {}
    score_inf_teory = {}
    time_hc_calculate = []

    time_both = {}
    score_both = {}
    time_data_both = []

    TARGET = config.label
    DRIVER = config.driver

    inf_handle = InformationHandle(label_feature_name=TARGET,
                                   driver_feature_name=DRIVER,
                                   dx=dx,
                                   dy=dy,
                                   taux=tx,
                                   tauy=ty)

    getdata = GetData('../../ThisCarIsMine', TARGET, DRIVER)

    mdebug(
        f'''
Sample size: {SAMPLE_SIZE}
HC size: {HC_SIZE}
Number of repetitions: {NUM_REPETITIONS}
dx,dy,tx,ty = {dx, dy, tx, ty}
'''
    )

    for i in range(NUM_REPETITIONS):
        for driver_target in ['A', 'B', 'C', 'D']:
            df_raw = getdata.get_sample(SAMPLE_SIZE,
                                        driver_target=driver_target)
            mdebug('Dataset size shape:', df_raw.shape)
            mdebug('\nDriver', driver_target, '\n')

            # Data process

            # Literature
            pp = LiteraturePreprocessing(df_raw, TARGET, DRIVER)
            t0 = time.time()

            pp.remove_correlation()
            pp.remove_invariants()
            pp.normalization()
            pp.remove_miss_value()
            df_pp2 = pp.get_df()
            pp.window()

            df_pp = pp.get_df()
            time_dataprocessing.append(time.time() - t0)

            X, y = df_pp.drop([TARGET, DRIVER], axis=1), df_pp[TARGET]
            mdebug(f"Literature.", df_pp.drop([TARGET, DRIVER], axis=1).shape[1],
                   f'Features after pre-processing:',
                   list(df_pp.drop([TARGET, DRIVER], axis=1).columns))
            score_literature, time_literature = classifiers_train_test(X, y,
                                                                       score_literature,
                                                                       time_literature,
                                                                       config.k_fold)

            # Entropy-Complexty (HC)
            mdebug(f"Information-Teory.")

            t0 = time.time()
            df_hc = inf_handle.get_information_binary_label(df_raw)
            time_hc_calculate.append(time.time() - t0)

            X, y = df_hc.drop([TARGET, DRIVER], axis=1), df_hc[TARGET]
            score_inf_teory, time_inf_teory = classifiers_train_test(X, y,
                                                                     score_inf_teory,
                                                                     time_inf_teory,
                                                                     config.k_fold)

            # Literature + Entropy-Complexty (HC)
            mdebug(f"Both.")

            df_joined = inf_handle.get_information_binary_label(df_pp2)

            X, y = df_joined.drop([TARGET, DRIVER], axis=1), df_joined[TARGET]
            mdebug(f"Literature + Entropy-Complexty.", df_joined.drop([TARGET, DRIVER], axis=1).shape[1],
                   f'Features after pre-processing:',
                   list(df_joined.drop([TARGET, DRIVER], axis=1).columns))
            score_both, time_both = classifiers_train_test(X, y,
                                                           score_both,
                                                           time_both,
                                                           config.k_fold)

    # Chart of score

    y_lt_accuracy = []
    y_lt_std_error_accuracy = []
    y_inf_accuracy = []
    y_inf_std_error_accuracy = []
    y_both_accuracy = []
    y_both_std_error_accuracy = []

    y_lt_trainingtime = []
    y_lt_std_error_trainingtime = []
    y_inf_trainingtime = []
    y_inf_std_error_trainingtime = []
    y_both_trainingtime = []
    y_both_std_error_trainingtime = []

    y_lt_totaltime = []
    y_lt_std_error_totaltime = []
    y_inf_totaltime = []
    y_inf_std_error_totaltime = []
    y_both_totaltime = []
    y_both_std_error_totaltime = []

    mdebug('\nChart')

    for key in classifier_names:
        y_lt_accuracy.append(mean(score_literature[key]))
        y_lt_std_error_accuracy.append(sem(score_literature[key]))

        y_inf_accuracy.append(mean(score_inf_teory[key]))
        y_inf_std_error_accuracy.append(sem(score_inf_teory[key]))

        y_both_accuracy.append(mean(score_both[key]))
        y_both_std_error_accuracy.append(sem(score_both[key]))

    X_axis = np.arange(len(classifier_names))
    mdebug(f'''
[data]  Classifier's Accuracy:

- Literature:
  - accuracy:  {y_lt_accuracy}
  - std error: {y_lt_std_error_accuracy}

- Entropy-Complexty:
  - accuracy:  {y_inf_accuracy}
  - std error: {y_inf_std_error_accuracy}

- Both:
  - accuracy:  {y_both_accuracy}
  - std error: {y_both_std_error_accuracy}

[end data]
        ''')

    mdebug('Output training times (ms):')
    for key in classifier_names:
        y_lt_trainingtime.append(mean(time_literature[key]) * 1000)
        y_lt_std_error_trainingtime.append(sem(time_literature[key]) * 1000)

        y_inf_trainingtime.append(mean(time_inf_teory[key]) * 1000)
        y_inf_std_error_trainingtime.append(sem(time_inf_teory[key]) * 1000)

        y_both_trainingtime.append(mean(time_both[key]) * 1000)
        y_both_std_error_trainingtime.append(sem(time_both[key]) * 1000)

        mdebug(f'''
    {key}:
      - Literature       : {y_lt_trainingtime[-1]} +- {y_lt_std_error_trainingtime[-1]}
      - Entropy-Complexty: {y_inf_trainingtime[-1]} +- {y_inf_std_error_trainingtime[-1]}
      - Both             : {y_both_trainingtime[-1]} +- {y_both_std_error_trainingtime[-1]}
''')

    mdebug(f'''
[data]  Classifier's Training time (ms):

- Literature:
  y_lt_trainingtime = {y_lt_trainingtime}
  y_lt_std_error_trainingtime = {y_lt_std_error_trainingtime}
  y_inf_trainingtime = {y_inf_trainingtime}
  y_inf_std_error_trainingtime = {y_inf_std_error_trainingtime}
  y_both = {y_both_trainingtime}
  y_both_std_error_trainingtime = {y_both_std_error_trainingtime}

[end data]
        ''')

    mdebug(f'''
[data]  Classifier's Data handle time (ms):

  time_dataprocessing = {time_dataprocessing}
  time_hc_calculate = {time_hc_calculate}

[end data]
        ''')

    mdebug('Output total times (ms):')
    mean_time_dataprocessing = mean(time_dataprocessing)
    mean_time_hc_calculate = mean(time_hc_calculate)
    mean_both = mean_time_dataprocessing + mean_time_hc_calculate
    for key in classifier_names:
        y_lt_totaltime.append((mean(time_literature[key]) + mean_time_dataprocessing) * 1000)
        y_lt_std_error_totaltime.append(sem(time_literature[key]) * 1000)

        y_inf_totaltime.append((mean(time_inf_teory[key]) + mean_time_hc_calculate) * 1000)
        y_inf_std_error_totaltime.append(sem(time_inf_teory[key]) * 1000)

        y_both_totaltime.append((mean(time_both[key]) + mean_both) * 1000)
        y_both_std_error_totaltime.append(sem(time_both[key]) * 1000)

        mdebug(f'''
    {key}:
      - Literature :        {y_lt_totaltime[-1]} +- {y_lt_std_error_totaltime[-1]}
      - Entropy-Complexty : {y_inf_totaltime[-1]} +- {y_inf_std_error_totaltime[-1]}
      - Both :              {y_both_totaltime[-1]} +- {y_both_std_error_totaltime[-1]}
''')

    mdebug(f'''
[data]  Classifier's Total time (ms):

  y_lt_totaltime = {y_lt_totaltime}
  y_lt_std_error_totaltime = {y_lt_std_error_totaltime}
  y_inf_totaltime = {y_inf_totaltime}
  y_inf_std_error_totaltime =  {y_inf_std_error_totaltime}
  y_both_totaltime = {y_both_totaltime}
  y_both_std_error_totaltime = {y_both_std_error_totaltime}

[end data]
        ''')

    pltr = PlotResults(classifier_names, SAMPLE_SIZE, output_folder='results/classifier_analysis')
    pltr.plot_accuracy(y_lt_accuracy, y_lt_std_error_accuracy,
                       y_inf_accuracy, y_inf_std_error_accuracy,
                       y_both_accuracy, y_both_std_error_accuracy)
    pltr.plot_trainingtime(y_lt_trainingtime, y_lt_std_error_trainingtime,
                           y_inf_trainingtime, y_inf_std_error_trainingtime,
                           y_both_trainingtime, y_both_std_error_trainingtime)
    pltr.plot_processingtime(time_dataprocessing, time_hc_calculate)
    pltr.plot_totaltime(y_lt_totaltime, y_lt_std_error_totaltime,
                        y_inf_totaltime, y_inf_std_error_totaltime,
                        y_both_totaltime, y_both_std_error_totaltime)

    delta = str(timedelta(seconds=time.time() - program_time))
    mdebug(f'\n[time] {delta}')
    print('\n[time]', delta)

    plt.show()
