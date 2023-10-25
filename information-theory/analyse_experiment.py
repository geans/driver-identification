#!/usr/bin/python3
import json
import time

import pandas as pd
import sklearn

import config
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
import warnings

warnings.filterwarnings("ignore")

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

f_without_na = ['calculation_overhead_entropy', 'calculation_overhead_complexity',
                'current_fire_timing_entropy', 'current_fire_timing_complexity',
                'cooling_temperature_entropy', 'cooling_temperature_complexity',
                'engine_speed_entropy', 'engine_speed_complexity']


def plot_experiment(score_lit, fit_time_lit, score_inf, fit_time_inf, ylim=(0.5, 1.05)):
    color1 = '#98c1d9'
    color2 = '#6495ed'
    color3 = '#0000ff'
    font_size = 24
    bottom_size = 0.3
    plt.figure('Score', figsize=(17, 9))
    X_axis = np.arange(len(classifier_names))
    plt.bar(X_axis - 0.2, [mean(value) for value in score_lit.values()], 0.4,
            yerr=[sem(value) for value in score_lit.values()], label='Literature', color=color1)
    plt.bar(X_axis + 0.2, [mean(value) for value in score_inf.values()], 0.4,
            yerr=[sem(value) for value in score_inf.values()], label='Proposal', color=color2)
    plt.ylim(ylim)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)

    plt.figure('Time', figsize=(17, 9))
    plt.bar(X_axis - 0.2, [mean(value) for value in fit_time_lit.values()], 0.4,
            yerr=[sem(value) for value in fit_time_lit.values()], label='Literature', color=color1)
    plt.bar(X_axis + 0.2, [mean(value) for value in fit_time_inf.values()], 0.4,
            yerr=[sem(value) for value in fit_time_inf.values()], label='Proposal', color=color2)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)

    plt.show()


def experiment_1_measure(window_size, k_fold, path, feature):
    def classifier_handle():
        try:
            result_dict = cross_validate(clf, X, y, cv=k_fold, return_estimator=True, return_train_score=True)
            score[clf_name] = score.get(clf_name, []) + result_dict['test_score'].tolist()
            fit_time[clf_name] = fit_time.get(clf_name, []) + result_dict['fit_time'].tolist()
        except Exception as e:
            print('Error in "classifier_handle".', e)

    class_feat = 'driver'
    # TODO: add random to choise file
    df_arr = []
    limit = 99999
    for i, driver in enumerate('ABCD'):
        df_arr += [pd.read_csv(f'{path}/{driver}/All_1.csv')]
        limit = min(limit, df_arr[i].shape[0])
    limit = 500
    print('limit =', limit)
    for i, driver in enumerate('ABCD'):
        df_arr[i] = df_arr[i][:limit]
        df_arr[i][class_feat] = [driver] * df_arr[i].shape[0]
    manager = multiprocessing.Manager()
    score = manager.dict()
    fit_time = manager.dict()
    # Split dataset into windows
    sliding_window = [pd.concat(
        [df_arr[0][i:i + window_size], df_arr[1][i:i + window_size], df_arr[2][i:i + window_size],
         df_arr[3][i:i + window_size]]) for i in range(0, limit - window_size, window_size // 2)]
    total_process = len(sliding_window) * len(classifier_names) * len('ABCD')
    print('total_process =', total_process)
    counter = 0
    # Process each window
    for window in sliding_window:
        X = window.drop([class_feat], axis=1)[feature]
        X = X.dropna(axis=1)
        running_process = []
        for clf, clf_name in zip(classifiers, classifier_names):
            for driver in 'ABCD':
                y = window[class_feat].replace(['A', 'B', 'C', 'D'],
                                               ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
                p = multiprocessing.Process(target=classifier_handle,
                                            args=())
                p.start()
                running_process.append(p)
        for p in running_process:
            p.join()
            counter += 1
            print(f'{counter}/{total_process}')
    return score, fit_time


def experiment_2_measure(path, feature):
    def classifier_handle():
        try:
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_time[clf_name] = fit_time.get(clf_name, []) + [time.time() - t0]
            y_pred = clf.predict(X_test)
            score_value = sklearn.metrics.accuracy_score(y_pred, y_test)
            score[clf_name] = score.get(clf_name, []) + [score_value]
        except Exception as e:
            print('Error in "classifier_handle".', e)

    class_feat = 'driver'
    # TODO: add random to choise file
    df_arr = []
    df_test_arr = []
    for i, driver in enumerate('ABCD'):
        df_arr += [pd.read_csv(f'{path}/{driver}/All_1.csv')[feature].dropna()]
        df_test_arr += [pd.concat([pd.read_csv(f'{path}/{driver}/All_2.csv')[feature].dropna(),
                                   pd.read_csv(f'{path}/{driver}/All_3.csv')[feature].dropna(),
                                   pd.read_csv(f'{path}/{driver}/All_4.csv')[feature].dropna(),
                                   pd.read_csv(f'{path}/{driver}/All_5.csv')[feature].dropna()])]
    for i, driver in enumerate('ABCD'):
        df_arr[i][class_feat] = [driver] * df_arr[i].shape[0]
        df_test_arr[i][class_feat] = [driver] * df_test_arr[i].shape[0]
    manager = multiprocessing.Manager()
    score = manager.dict()
    fit_time = manager.dict()
    counter = 0
    # Train data
    df = pd.concat(df_arr)
    print('df.shape =', df.shape)
    X_train = df.drop([class_feat], axis=1)
    # Test data
    df_test = pd.concat(df_test_arr)
    print('df_test.shape =', df_test.shape)
    X_test = df_test.drop([class_feat], axis=1)
    # Process
    running_process = []
    total_process = len(classifier_names) * 4
    for clf, clf_name in zip(classifiers, classifier_names):
        for driver in 'ABCD':
            y_train = df[class_feat].replace(['A', 'B', 'C', 'D'],
                                             ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
            y_test = df_test[class_feat].replace(['A', 'B', 'C', 'D'],
                                                 ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
            p = multiprocessing.Process(target=classifier_handle,
                                        args=())
            p.start()
            running_process.append(p)
    for p in running_process:
        p.join()
        counter += 1
        print(f'{counter}/{total_process}')
    return score, fit_time


def experiment_3_measure(path, feature):
    def classifier_handle():
        try:
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_time[clf_name] = fit_time.get(clf_name, []) + [time.time() - t0]
            y_pred = clf.predict(X_test)
            score_value = sklearn.metrics.accuracy_score(y_pred, y_test)
            score[clf_name] = score.get(clf_name, []) + [score_value]
        except Exception as e:
            print('Error in "classifier_handle".', e)

    class_feat = 'driver'
    # TODO: add random to choise file
    df_arr = []
    df_test_arr = []
    for i, driver in enumerate('ABCD'):
        df_arr += [pd.read_csv(f'{path}/{driver}/All_1.csv')[feature].dropna()]
        df_test_arr += [pd.concat([pd.read_csv(f'{path}/{driver}/All_2.csv')[feature].dropna(),
                                   pd.read_csv(f'{path}/{driver}/All_3.csv')[feature].dropna(),
                                   pd.read_csv(f'{path}/{driver}/All_4.csv')[feature].dropna(),
                                   pd.read_csv(f'{path}/{driver}/All_5.csv')[feature].dropna()])]
    for i, driver in enumerate('ABCD'):
        df_arr[i][class_feat] = [driver] * df_arr[i].shape[0]
        df_test_arr[i][class_feat] = [driver] * df_test_arr[i].shape[0]
    manager = multiprocessing.Manager()
    score = manager.dict()
    fit_time = manager.dict()
    counter = 0
    # Train data
    df = pd.concat(df_arr)
    print('df.shape =', df.shape)
    X_train = df.drop([class_feat], axis=1)
    # Test data
    df_test = pd.concat(df_test_arr)
    print('df_test.shape =', df_test.shape)
    X_test = df_test.drop([class_feat], axis=1)
    # Process
    running_process = []
    total_process = len(classifier_names)
    for clf, clf_name in zip(classifiers, classifier_names):
        y_train = df[class_feat]
        y_test = df_test[class_feat]
        p = multiprocessing.Process(target=classifier_handle,
                                    args=())
        p.start()
        running_process.append(p)
    for p in running_process:
        p.join()
        counter += 1
        print(f'{counter}/{total_process}')
    return score, fit_time


def experiment_1():
    print('Running experiment 1')
    # score_lit, fit_time_lit = experiment_1_measure(120, 5, '../../ThisCarIsMineInf', config.feature_lit)
    # score_inf, fit_time_inf = experiment_1_measure(120, 5, '../../ThisCarIsMineInf', config.feature_inf)
    # with open('analyse_experiment_1.out_values.txt', 'w') as out:
    #     json.dump([score_lit.copy(), fit_time_lit.copy(), score_inf.copy(), fit_time_inf.copy()], out)

    with open('analyse_experiment_1.out_values.txt', 'r') as data_file:
        data = json.load(data_file)
        plot_experiment(*data)


def experiment_2():
    print('Running experiment 2')
    # score_lit, fit_time_lit = experiment_2_measure('../../ThisCarIsMineInf', config.feature_lit)
    # score_inf, fit_time_inf = experiment_2_measure('../../ThisCarIsMineInf', config.feature_inf)
    # with open('analyse_experiment_2.out_values.txt', 'w') as out:
    #     json.dump([score_lit.copy(), fit_time_lit.copy(), score_inf.copy(), fit_time_inf.copy()], out)

    with open('analyse_experiment_2.out_values.txt', 'r') as data_file:
        data = json.load(data_file)
        plot_experiment(*data)


def experiment_3():
    print('Running experiment 3')
    # score_lit, fit_time_lit = experiment_3_measure('../../ThisCarIsMineInf', config.feature_lit)
    # score_inf, fit_time_inf = experiment_3_measure('../../ThisCarIsMineInf', config.feature_inf)
    # with open('analyse_experiment_3.out_values.txt', 'w') as out:
    #     json.dump([score_lit.copy(), fit_time_lit.copy(), score_inf.copy(), fit_time_inf.copy()], out)

    with open('analyse_experiment_3.out_values.txt', 'r') as data_file:
        data = json.load(data_file)
        plot_experiment(*data, ylim=(0.0, 0.55))


if __name__ == '__main__':
    experiment_1()
    experiment_2()
    experiment_3()
