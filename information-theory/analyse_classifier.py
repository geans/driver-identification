#!/usr/bin/python3

import config
from plot_results import PlotResults
from datetime import timedelta
from getdata import GetData
from information import InformationHandle
from numpy import mean
from scipy.stats import sem
from sklearn import metrics
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
import analyse_data

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


def get_metrics(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    return accuracy


def test_classifier(clf, data_size_fit):
    mdebug('\n[Test-Zone]\n')
    TARGET = config.label
    DRIVER = config.driver
    data_full = GetData(path_dataset='../../ThisCarIsMine',
                        label_feature_name=config.label,
                        driver_feature_name=config.driver,
                        features=config.ALL_FEATURES,
                        trips=None)

    data_to_fit = GetData(path_dataset='../../ThisCarIsMine',
                        label_feature_name=config.label,
                        driver_feature_name=config.driver,
                          trips=[1, 2, 3, 4])

    data_to_test = GetData(path_dataset='../../ThisCarIsMine',
                        label_feature_name=config.label,
                        driver_feature_name=config.driver,
                           trips=[5])

    # analyse data
    df = data_full.get_all().drop([TARGET, DRIVER], axis=1)
    invariance, variance_df = analyse_data.analyse_variance(df)
    corr, included, excluded = analyse_data.analyse_correlation(df=variance_df, correlate_threshold=.95)


    inf_handle = InformationHandle(label_feature_name=TARGET,
                                   driver_feature_name=DRIVER,
                                   dx=dx,
                                   dy=dy,
                                   taux=tx,
                                   tauy=ty)

    for i in range(1):
        for driver_target in ['A', 'B', 'C', 'D']:
            df_fit = data_to_fit.get_sample(sample_size=data_size_fit,
                                            driver_target=driver_target)[included + [TARGET, DRIVER]]
            df_test = data_to_test.get_all(driver_target=driver_target)[included + [TARGET, DRIVER]]

            # Literature
            pp = LiteraturePreprocessing(df_fit, TARGET, DRIVER)
            pp.normalization()
            pp.remove_miss_value()
            df_pp2 = pp.get_df()
            pp.window()
            df_pp = pp.get_df()
            #
            pp_test = LiteraturePreprocessing(df_test, TARGET, DRIVER)
            pp_test.normalization()
            pp_test.remove_miss_value()
            df_pp2_test = pp_test.get_df()
            pp_test.window()
            df_pp_test = pp_test.get_df()

            X_train, y_train = df_pp.drop([TARGET, DRIVER], axis=1), df_pp[TARGET]
            X_test, y_test = df_pp_test.drop([TARGET, DRIVER], axis=1), df_pp_test[TARGET]
            mdebug(f"Literature.")
            accuracy_lt = get_metrics(clf, X_train, y_train, X_test, y_test)

            # Entropy-Complexty (HC)
            mdebug(f"Information-Teory.")

            df_hc = inf_handle.get_information_binary_label(df_fit)
            df_hc_test = inf_handle.get_information_binary_label(df_test)

            X_train, y_train = df_hc.drop([TARGET, DRIVER], axis=1), df_hc[TARGET]
            X_test, y_test = df_hc_test.drop([TARGET, DRIVER], axis=1), df_hc_test[TARGET]
            accuracy_inf = get_metrics(clf, X_train, y_train, X_test, y_test)

            # Literature + Entropy-Complexty (HC)
            mdebug(f"Both.")

            df_joined = inf_handle.get_information_binary_label(df_pp2)
            df_joined_test = inf_handle.get_information_binary_label(df_pp2_test)

            X_train, y_train = df_joined.drop([TARGET, DRIVER], axis=1), df_joined[TARGET]
            X_test, y_test = df_joined_test.drop([TARGET, DRIVER], axis=1), df_joined_test[TARGET]
            accuracy_both = get_metrics(clf, X_train, y_train, X_test, y_test)

            bar_width = 0.3
            X_axis = np.arange(3)
            fig, ax = plt.subplots(1, 1, figsize=(17, 9))
            ax.bar(X_axis - bar_width, [accuracy_lt, accuracy_inf, accuracy_both])
            ax.bar(X_axis, accuracy_inf)
            ax.bar(X_axis + bar_width, accuracy_both)

    mdebug('\n[End-Test-Zone]\n')


if __name__ == '__main__':
    # test_classifier(SVC(gamma=.9, C=1), 50)
    # plt.show()
    # exit()

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

    getdata_handler = GetData(path_dataset='../../ThisCarIsMine',
                              label_feature_name=TARGET,
                              driver_feature_name=DRIVER,
                              features=config.ALL_FEATURES,
                              trips=[1, 2, 3, 4])

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
            df_raw = getdata_handler.get_sample(SAMPLE_SIZE,
                                                driver_target=driver_target)
            mdebug('Dataset size shape:', df_raw.shape)
            mdebug('\nDriver', driver_target, '\n')

            # Data process

            # Literature
            pp = LiteraturePreprocessing(df_raw, TARGET, DRIVER)
            t0 = time.time()

            pp.remove_correlation(.95)
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
