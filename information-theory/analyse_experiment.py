#!/usr/bin/python3
import json
import time

import pandas as pd
import sklearn
from sklearn import metrics

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
    font_size = 24
    bottom_size = 0.3
    X_axis = np.arange(len(classifier_names))

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

    plt.figure('Score', figsize=(17, 9))
    x_lit_score = [mean(value) for value in score_lit.values()]
    x_inf_score = [mean(value) for value in score_inf.values()]
    print('Literature', min(x_lit_score), max(x_lit_score))
    print('Proposal', min(x_inf_score), max(x_inf_score))
    plt.bar(X_axis - 0.2, x_lit_score, 0.4,
            yerr=[sem(value) for value in score_lit.values()], label='Literature', color=color1)
    plt.bar(X_axis + 0.2, x_inf_score, 0.4,
            yerr=[sem(value) for value in score_inf.values()], label='Proposal', color=color2)
    plt.ylim(ylim)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)

    plt.show()


def plot_roc(y_test_lit, y_prediction_lit, y_test_inf, y_prediction_inf):
    color1 = '#98c1d9'
    color2 = '#6495ed'
    fpr_lit, tpr_lit, threshold_lit = metrics.roc_curve(y_test_lit, y_prediction_lit)
    roc_auc_lit = metrics.auc(fpr_lit, tpr_lit)
    fpr_inf, tpr_inf, threshold_inf = metrics.roc_curve(y_test_inf, y_prediction_inf)
    roc_auc_inf = metrics.auc(fpr_inf, tpr_inf)

    # method I: plt
    plt.figure('Receiver Operating Characteristic')
    plt.plot(fpr_lit, tpr_lit, 'b', label=f'Literature, AUC = {roc_auc_lit:0.2f}', color=color1)
    plt.plot(fpr_inf, tpr_inf, 'b', label=f'Proposal, AUC = {roc_auc_inf:0.2f}', color=color2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
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
    limit = 300
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
    def classifier_handle(y_test_arr, y_prediction_arr):
        try:
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_time[clf_name] = fit_time.get(clf_name, []) + [time.time() - t0]
            y_prediction = clf.predict(X_test)
            score_value = sklearn.metrics.accuracy_score(y_prediction, y_test)
            score[clf_name] = score.get(clf_name, []) + [score_value]
            y_test_arr += y_test.tolist()
            y_prediction_arr += y_prediction.tolist()
        except Exception as e:
            print('Error in "classifier_handle".', e)

    class_feat = 'driver'
    # TODO: add random to choise file
    df_arr = []
    df_test_arr = []
    for i, driver in enumerate('ABCD'):
        df_arr += [pd.read_csv(f'{path}/{driver}/All_1.csv')[feature].dropna()]
        df_test_arr += [pd.concat([pd.read_csv(f'{path}/{driver}/All_2.csv')[feature].dropna(),
                                   # pd.read_csv(f'{path}/{driver}/All_3.csv')[feature].dropna(),
                                   # pd.read_csv(f'{path}/{driver}/All_4.csv')[feature].dropna(),
                                   # pd.read_csv(f'{path}/{driver}/All_5.csv')[feature].dropna()
                                   ])]
    for i, driver in enumerate('ABCD'):
        df_arr[i][class_feat] = [driver] * df_arr[i].shape[0]
        df_test_arr[i][class_feat] = [driver] * df_test_arr[i].shape[0]
    manager = multiprocessing.Manager()
    score = manager.dict()
    fit_time = manager.dict()
    y_test_arr = manager.list()
    y_prediction_arr = manager.list()
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
                                        args=(y_test_arr, y_prediction_arr))
            p.start()
            running_process.append(p)
    for p in running_process:
        p.join()
        counter += 1
        print(f'{counter}/{total_process}')
    return score.copy(), fit_time.copy(), list(y_test_arr), list(y_prediction_arr)


def experiment_sample_size_measure(path, feature):
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

    score_by_sample_size = []

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
    _df = pd.concat(df_arr)
    # Test data
    df_test = pd.concat(df_test_arr)
    X_test = df_test.drop([class_feat], axis=1)
    manager = multiprocessing.Manager()
    for sample_size in range(10, 510, 100):
        score = manager.dict()
        fit_time = manager.dict()
        counter = 0
        # Train data
        df = _df[:sample_size]
        X_train = df.drop([class_feat], axis=1)
        # Process
        running_process = []
        total_process = 4
        for clf, clf_name in zip([RandomForestClassifier(max_depth=51, n_estimators=10, max_features=10)],
                                 ['R. Forest']):
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
            print(f'{counter}/{total_process}', end='\r')
        print('\nsample_size =', sample_size)
        print('score =', mean(score['R. Forest']), sem(score['R. Forest']))
        print('time =', mean(fit_time['R. Forest']), sem(fit_time['R. Forest']))
        score_by_sample_size.append((sample_size,
                                     mean(score['R. Forest']), sem(score['R. Forest']),
                                     mean(fit_time['R. Forest']), sem(fit_time['R. Forest'])))
    return score_by_sample_size


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
    score_lit, fit_time_lit = experiment_1_measure(120, 5, '../../ThisCarIsMine', config.feature_lit)
    score_inf, fit_time_inf = experiment_1_measure(120, 5, '../../ThisCarIsMineInf_window300_dx6', config.feature_inf)
    with open('analyse_experiment_1.out_values.txt', 'w') as out:
        json.dump([score_lit.copy(), fit_time_lit.copy(), score_inf.copy(), fit_time_inf.copy()], out)

    with open('analyse_experiment_1.out_values.txt', 'r') as data_file:
        data = json.load(data_file)
        plot_experiment(*data)


def experiment_2():
    print('Running experiment 2')
    data_lit = experiment_2_measure('../../ThisCarIsMineInf', config.feature_lit)
    data_inf = experiment_2_measure('../../ThisCarIsMineInf', config.feature_inf)
    with open('analyse_experiment_2.out_values.txt', 'w') as out:
        json.dump((data_lit, data_inf), out)

    with open('analyse_experiment_2.out_values.txt', 'r') as data_file:
        data_lit, data_inf = json.load(data_file)
        score_lit, fit_time_lit, y_test_arr_lit, y_prediction_arr_lit = data_lit
        score_inf, fit_time_inf, y_test_arr_inf, y_prediction_arr_inf = data_inf
        # Plot bars
        plot_experiment(score_lit, fit_time_lit, score_inf, fit_time_inf)
        # Plot roc
        plot_roc(y_test_arr_lit, y_prediction_arr_lit, y_test_arr_inf, y_prediction_arr_inf)


def experiment_sample_size():
    print('Running experiment 2.1')

    score_by_sample_size_lit = experiment_sample_size_measure('../../ThisCarIsMineInf', config.feature_lit)
    score_by_sample_size_inf = experiment_sample_size_measure('../../ThisCarIsMineInf', config.feature_inf)
    with open('analyse_experiment_2_1.out_values.txt', 'w') as out:
        json.dump([score_by_sample_size_lit, score_by_sample_size_inf], out)
    # exit()

    with open('analyse_experiment_2_1.out_values.txt', 'r') as data_file:
        score_by_sample_size_lit, score_by_sample_size_inf = json.load(data_file)
        color1 = '#98c1d9'
        color2 = '#6495ed'
        color3 = '#0000ff'
        ylim = (0, 1.05)
        font_size = 24
        bottom_size = 0.3
        plt.figure('Score', figsize=(17, 9))

        x, y_lit, y_inf, y_lit_err, y_inf_err = [], [], [], [], []
        for _x, _y_lit, _y_lit_err, _, __ in score_by_sample_size_lit:
            x.append(_x)
            y_lit.append(_y_lit)
            y_lit_err.append(_y_lit_err)
        for _x, _y_inf, _y_inf_err, _, __ in score_by_sample_size_inf:
            y_inf.append(_y_inf)
            y_inf_err.append(_y_inf_err)
        print('x =', x)
        print('y_lit =', y_lit)
        print('y_lit_err =', y_lit_err)
        print('y_inf =', y_inf)
        print('y_inf_err =', y_inf_err)

        plt.errorbar(x, y_lit, yerr=y_lit_err, label='Literature', color=color1)
        plt.errorbar(x, y_inf, yerr=y_inf_err, label='Proposal', color=color2)
        plt.ylim(ylim)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
                   fancybox=True, shadow=True, ncol=3, fontsize=font_size)
        plt.subplots_adjust(bottom=bottom_size)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.tick_params(axis='x', labelrotation=0)

        plt.show()


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
    # experiment_1()
    experiment_2()
    # experiment_3()
    # experiment_sample_size()
