#!/usr/bin/python3
import json
import math
import random
import time
from random import randint

import pandas as pd
import sklearn
from sklearn import metrics

import config
from numpy import mean
from scipy.stats import sem
from sklearn.model_selection import cross_val_predict
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

color1 = '#115f9a'
color2 = '#009dc3'
color3 = '#00d38d'
color4 = '#d0f400'


def my_debug(*objects, sep=' ', end='\n', file=None, flush=False, path='.'):
    if config.debug_on_screen:
        print(*objects, sep=sep, end=end, file=file, flush=flush)
    # output_file = open(f'{path}/log.txt', 'a')
    # for obj in objects:
    #     output_file.write(str(obj))
    #     output_file.write(sep)
    # output_file.write(end)
    # output_file.close()


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
    KNeighborsClassifier(math.floor(math.sqrt(config.inf_window_size))),
    SVC(kernel="linear"),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    GaussianNB(),
]


def plot_experiment(score_lit, score_inf, score_name, ylim=(0.5, 1.05), fig_name='', path=None):
    font_size = 24
    bottom_size = 0.3
    X_axis = np.arange(len(classifier_names))

    plt.figure('Score. ' + fig_name, figsize=config.default_figsize)
    x_lit_score = [mean(value) for value in score_lit.values()]
    x_inf_score = [mean(value) for value in score_inf.values()]
    my_debug(f'{score_name}, length:', len(list(score_lit.values())[0]))
    my_debug('  Literature', min(x_lit_score), 'to', max(x_lit_score))
    my_debug('  Proposal', min(x_inf_score), 'to', max(x_inf_score))
    plt.bar(X_axis - 0.2, x_lit_score, 0.4,
            yerr=[sem(value) for value in score_lit.values()], label='Literature', color=color1, edgecolor="black")
    plt.bar(X_axis + 0.2, x_inf_score, 0.4,
            yerr=[sem(value) for value in score_inf.values()], label='Proposal', color=color3, edgecolor="black")
    plt.ylim(ylim)
    plt.ylabel(f'{score_name} Score', fontsize=font_size)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)
    # plt.spines['top'].set_visible(False)
    # plt.spines['right'].set_visible(False)
    # plt.spines['bottom'].set_visible(False)
    # plt.spines['left'].set_visible(False)
    if path is not None:
        plt.savefig(f'{path}/{fig_name}.png')


def plot_roc(y_test_lit, y_pred_lit, y_test_inf, y_pred_inf, fig_name='', path=None):
    y_min = min(len(y_test_lit), len(y_pred_lit), len(y_test_inf), len(y_pred_inf))

    y_test_lit, y_pred_lit = y_test_lit[:y_min], y_pred_lit[:y_min]
    fpr_lit, tpr_lit, threshold_lit = metrics.roc_curve(y_test_lit, y_pred_lit)
    roc_auc_lit = metrics.auc(fpr_lit, tpr_lit)

    y_test_inf, y_pred_inf = y_test_inf[:y_min], y_pred_inf[:y_min]
    fpr_inf, tpr_inf, threshold_inf = metrics.roc_curve(y_test_inf, y_pred_inf)
    roc_auc_inf = metrics.auc(fpr_inf, tpr_inf)

    plt.figure('ROC Curve. ' + fig_name)
    plt.plot(fpr_lit, tpr_lit, 'b', label=f'Literature, AUC = {roc_auc_lit:0.2f}', color=color1)
    plt.plot(fpr_inf, tpr_inf, 'b', label=f'Proposal, AUC = {roc_auc_inf:0.2f}', color=color3)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if path is not None:
        plt.savefig(f'{path}/ROC_{fig_name}.png')
    return roc_auc_lit, roc_auc_inf


def plot_experiment_4bars(score_lit, score_inf_hc, score_inf_fs, score_inf_hcfs, score_name, ylim=(0.5, 1.05),
                          fig_name='', path=None):
    font_size = 24
    legend_font_size = 18
    bottom_size = 0.27
    X_axis = np.arange(len(classifier_names))
    bar_width = 0.2

    plt.figure(f'{score_name} Score. {fig_name}', figsize=config.default_figsize)
    x_lit_score = [mean(value) for value in score_lit.values()]
    x_inf_hc_score = [mean(value) for value in score_inf_hc.values()]
    x_inf_fs_score = [mean(value) for value in score_inf_fs.values()]
    x_inf_hc_fs_score = [mean(value) for value in score_inf_hcfs.values()]
    my_debug(f'{score_name}, length:', len(list(score_lit.values())[0]))
    my_debug('Literature', min(x_lit_score), max(x_lit_score))
    my_debug('Proposal HC', min(x_inf_hc_score), max(x_inf_hc_score))
    my_debug('Proposal FS', min(x_inf_fs_score), max(x_inf_fs_score))
    my_debug('Proposal HC+FS', min(x_inf_hc_fs_score), max(x_inf_hc_fs_score))
    plt.bar(X_axis - 0.3, x_lit_score, bar_width,
            yerr=[sem(value) for value in score_lit.values()],
            label='Literature', color=color1, edgecolor="black")
    plt.bar(X_axis - 0.1, x_inf_hc_score, bar_width,
            yerr=[sem(value) for value in score_inf_hc.values()],
            label='$\mathcal{H}\mathcal{C}$', color=color2, edgecolor="black")
    plt.bar(X_axis + 0.1, x_inf_hc_score, bar_width,
            yerr=[sem(value) for value in score_inf_fs.values()],
            label='$FS$', color=color3, edgecolor="black")
    plt.bar(X_axis + 0.3, x_inf_hc_fs_score, bar_width,
            yerr=[sem(value) for value in score_inf_hcfs.values()],
            label='$\mathcal{H}\mathcal{C}+FS$', color=color4, edgecolor="black")
    plt.ylim(ylim)
    plt.ylabel(f'{score_name} Score', fontsize=font_size)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22),
               fancybox=True, shadow=True, ncol=2, fontsize=legend_font_size)
    plt.subplots_adjust(bottom=bottom_size)
    # plt.legend(bbox_to_anchor=(1, .75),
    #            fancybox=True, shadow=True, ncol=1, fontsize=legend_font_size)
    # plt.subplots_adjust(right=.75, bottom=.3)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)

    plt.grid(axis='y')
    major_ticks = np.arange(.5, 1.01, .1)
    minor_ticks = np.arange(.5, 1.01, .05)
    plt.yticks(major_ticks)
    plt.yticks(minor_ticks, minor=True)
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.8)

    if path is not None:
        plt.savefig(f'{path}/{score_name}_score_results.png')


def plot_roc_auc_dict_4bars(roc_auc_lit, roc_auc_inf_hc, roc_auc_inf_fs, roc_auc_inf_hc_fs, ylim=(0.5, 1.05),
                            path=None):
    font_size = 24
    legend_font_size = 18
    bottom_size = 0.27
    X_axis = np.arange(len(classifier_names))
    bar_width = 0.2

    plt.figure('ROC AUC', figsize=config.default_figsize)
    x_lit = [mean(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    lit_err = [sem(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    x_inf_hc = [mean(roc_auc_inf_hc[clf_name]) for clf_name in classifier_names]
    inf_err_hc = [sem(roc_auc_inf_hc[clf_name]) for clf_name in classifier_names]
    x_inf_fs = [mean(roc_auc_inf_fs[clf_name]) for clf_name in classifier_names]
    inf_err_fs = [sem(roc_auc_inf_fs[clf_name]) for clf_name in classifier_names]
    x_inf_hc_fs = [mean(roc_auc_inf_hc_fs[clf_name]) for clf_name in classifier_names]
    inf_err_hc_fs = [sem(roc_auc_inf_hc_fs[clf_name]) for clf_name in classifier_names]
    my_debug('ROC AUC, ')
    my_debug('Literature', min(x_lit), max(x_lit))
    my_debug('Proposal HC', min(x_inf_hc), max(x_inf_hc))
    my_debug('Proposal FS', min(x_inf_fs), max(x_inf_fs))
    my_debug('Proposal HC + FS', min(x_inf_hc_fs), max(x_inf_hc_fs))
    plt.bar(X_axis - 0.3, x_lit, bar_width, yerr=lit_err, label='Literature', color=color1, edgecolor="black")
    plt.bar(X_axis - 0.1, x_inf_hc, bar_width, yerr=inf_err_hc, label='$\mathcal{H}\mathcal{C}$', color=color2, edgecolor="black")
    plt.bar(X_axis + 0.1, x_inf_fs, bar_width, yerr=inf_err_fs, label='$FS$', color=color3, edgecolor="black")
    plt.bar(X_axis + 0.3, x_inf_hc_fs, bar_width, yerr=inf_err_hc_fs, label='$\mathcal{H}\mathcal{C}+FS$', color=color4, edgecolor="black")
    plt.ylim(ylim)
    plt.ylabel('ROC AUC', fontsize=font_size)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22),
               fancybox=True, shadow=True, ncol=2, fontsize=legend_font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)

    plt.grid(axis='y')
    major_ticks = np.arange(.5, 1.01, .1)
    minor_ticks = np.arange(.5, 1.01, .05)
    plt.yticks(major_ticks)
    plt.yticks(minor_ticks, minor=True)
    plt.grid(which='minor', alpha=0.4)
    plt.grid(which='major', alpha=0.8)

    if path is not None:
        plt.savefig(f'{path}/roc_auc_mean.png')


def plot_roc_auc_dict(roc_auc_lit, roc_auc_inf, ylim=(0.5, 1.05), path=None):
    font_size = 24
    bottom_size = 0.6
    X_axis = np.arange(len(classifier_names))

    plt.figure('ROC AUC', figsize=config.default_figsize)
    x_lit = [mean(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    lit_err = [sem(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    x_inf = [mean(roc_auc_inf[clf_name]) for clf_name in classifier_names]
    inf_err = [sem(roc_auc_inf[clf_name]) for clf_name in classifier_names]
    my_debug('ROC AUC, ')
    my_debug('Literature', min(x_lit), max(x_lit))
    my_debug('Proposal', min(x_inf), max(x_inf))
    plt.bar(X_axis - 0.2, x_lit, 0.4, yerr=lit_err, label='Literature', color=color1, edgecolor="black")
    plt.bar(X_axis + 0.2, x_inf, 0.4, yerr=inf_err, label='Proposal', color=color3, edgecolor="black")
    plt.ylim(ylim)
    plt.ylabel('ROC AUC', fontsize=font_size)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)
    if path is not None:
        plt.savefig(f'{path}/roc_auc_mean.png')


def experiment_1_measure(window_size, k_fold, path, feature):
    def classifier_handle(score_dict, roc_auc_dict, precision_dict, recall_dict,
                          clf, clf_name, X, y, k_fold):
        try:
            y_pred = cross_val_predict(clf, X, y, cv=k_fold)
            fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
            roc_auc_dict[clf_name] = roc_auc_dict.get(clf_name, []) + [metrics.auc(fpr, tpr)]
            score_dict[clf_name] = score_dict.get(clf_name, []) + [metrics.accuracy_score(y, y_pred)]
            precision_dict[clf_name] = precision_dict.get(clf_name, []) + [metrics.precision_score(y, y_pred)]
            recall_dict[clf_name] = recall_dict.get(clf_name, []) + [metrics.recall_score(y, y_pred)]
        except Exception as e:
            my_debug(f'Error in "classifier_handle".', e, set(y), len(y))

    class_feat = 'driver'
    manager = multiprocessing.Manager()
    score_dict = manager.dict()
    roc_auc_dict = manager.dict()
    # y_dict = manager.dict()
    # y_pred_dict = manager.dict()
    precision_dict = manager.dict()
    recall_dict = manager.dict()
    df_arr = []
    for t in range(5):
        limit = 99999
        for driver, trips in zip('ABCD', (
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 6],
                [1, 2, 3, 4, 5],
                [1, 3, 5, 6, 7],
        )):
            trip = trips[t]
            csv_filename = f'{path}/{driver}/All_{trip}.csv'
            _df = pd.read_csv(csv_filename)
            df_arr += [_df[feature].dropna()]
            my_debug(csv_filename, df_arr[-1].shape)
            limit = min(limit, df_arr[-1].shape[0])
        # limit = 150
        my_debug('limit =', limit)
        for i, driver in enumerate('ABCD'):
            df_arr[i] = df_arr[i][:limit]
            df_arr[i][class_feat] = [driver] * df_arr[i].shape[0]
        # Split dataset into windows
        shift = window_size >> 1  # window_size // 2
        sliding_window = [pd.concat(
            [df_arr[0][j:j + window_size], df_arr[1][j:j + window_size],
             df_arr[2][j:j + window_size], df_arr[3][j:j + window_size]
             ]) for j in range(0, limit, shift)]
        total_process = len(sliding_window) * len(classifier_names) * len('ABCD')
        my_debug('total_process =', total_process)
        counter = 0
        # Process each window
        for window in sliding_window:
            if len(window) < window_size:
                my_debug('len(window)', len(window), 'window_size', window_size)
                my_debug('len(sliding_window)', len(sliding_window))
                continue
            X = window.drop([class_feat], axis=1)
            for clf, clf_name in zip(classifiers, classifier_names):
                running_process = []
                for driver in 'ABCD':
                    y = window[class_feat].replace(['A', 'B', 'C', 'D'],
                                                   ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
                    p = multiprocessing.Process(target=classifier_handle,
                                                args=(score_dict, roc_auc_dict, precision_dict, recall_dict,
                                                      clf, clf_name, X, y, k_fold))
                    p.start()
                    running_process.append(p)
                for p in running_process:
                    p.join()
                    counter += 1
                    my_debug(f'{t} | {counter}/{total_process}', end='\r')
        my_debug()
    return score_dict.copy(), roc_auc_dict.copy(), precision_dict.copy(), recall_dict.copy()


def experiment_2_measure(path, feature, path_to_save):
    def classifier_handle(score_dict, roc_auc_dict, clf, clf_name, X_train, y_train, X_test, y_test):
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score_value = metrics.accuracy_score(y_pred, y_test)
            score_dict[clf_name] = score_dict.get(clf_name, []) + [score_value]
            fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
            roc_auc_dict[clf_name] = roc_auc_dict.get(clf_name, []) + [metrics.auc(fpr, tpr)]
        except Exception as e:
            my_debug('Error in "classifier_handle".', e, path=path_to_save)

    class_feat = 'driver'
    total_process = len(classifier_names) * 4 * 5  # 4 = driver's amount, 5 = trip's amount
    counter = 0
    manager = multiprocessing.Manager()
    score = manager.dict()
    roc_auc = manager.dict()
    for repetition in range(5):
        my_debug(f'Repetition {repetition}', path=path_to_save)
        df_arr = []
        df_test_arr = []
        for driver, trips in zip('ABCD', (
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 6],
                [1, 2, 3, 4, 5],
                [1, 3, 5, 6, 7],
        )):
            trip = trips[repetition]
            df_test_arr += [pd.read_csv(f'{path}/{driver}/All_{trip}.csv')[feature].dropna()]
            trips.remove(trip)
            df_arr += [pd.concat([pd.read_csv(f'{path}/{driver}/All_{trip}.csv')[feature].dropna() for trip in trips])]
        for i, driver in enumerate('ABCD'):
            df_arr[i][class_feat] = [driver] * df_arr[i].shape[0]
            df_test_arr[i][class_feat] = [driver] * df_test_arr[i].shape[0]
        y_test_arr = manager.list()
        y_prediction_arr = manager.list()
        # Train data
        df_train = pd.concat(df_arr)
        my_debug('df_train.shape =', df_train.shape, path=path_to_save)
        X_train = df_train.drop([class_feat], axis=1)
        # Test data
        df_test = pd.concat(df_test_arr)
        my_debug('df_test.shape =', df_test.shape, path=path_to_save)
        X_test = df_test.drop([class_feat], axis=1)
        # Process
        running_process = []
        for clf, clf_name in zip(classifiers, classifier_names):
            for driver in 'ABCD':
                y_train = df_train[class_feat].replace(['A', 'B', 'C', 'D'],
                                                       ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
                y_test = df_test[class_feat].replace(['A', 'B', 'C', 'D'],
                                                     ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
                p = multiprocessing.Process(target=classifier_handle,
                                            args=(score, roc_auc, clf, clf_name, X_train, y_train, X_test, y_test))
                p.start()
                running_process.append(p)
        for p in running_process:
            p.join()
            counter += 1
            my_debug(f'{counter}/{total_process}', end='\r', path=path_to_save)
    my_debug(path=path_to_save)
    return score.copy(), roc_auc.copy()


def experiment_3_measure(path, feature):
    def classifier_handle(y_test_arr, y_prediction_arr):
        try:
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_time[clf_name] = fit_time.get(clf_name, []) + [time.time() - t0]
            y_pred = clf.predict(X_test)
            score_value = metrics.accuracy_score(y_pred, y_test)
            score[clf_name] = score.get(clf_name, []) + [score_value]
            y_test_arr += y_test.tolist()
            y_prediction_arr += y_pred.tolist()
        except Exception as e:
            my_debug('Error in "classifier_handle".', e)

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
    my_debug('df.shape =', df.shape)
    X_train = df.drop([class_feat], axis=1)
    # Test data
    df_test = pd.concat(df_test_arr)
    my_debug('df_test.shape =', df_test.shape)
    X_test = df_test.drop([class_feat], axis=1)
    # Process
    running_process = []
    total_process = len(classifier_names)
    for clf, clf_name in zip(classifiers, classifier_names):
        y_train = df[class_feat]
        y_test = df_test[class_feat]
        p = multiprocessing.Process(target=classifier_handle,
                                    args=(y_test_arr, y_prediction_arr))
        p.start()
        running_process.append(p)
    for p in running_process:
        p.join()
        counter += 1
        my_debug(f'{counter}/{total_process}')
    return score.copy(), fit_time.copy(), list(y_test_arr), list(y_prediction_arr)


def experiment_1():
    experiment_name = 'Experiment 1'
    my_debug('Running', experiment_name)
    path_to_save = './results/experiment_1'

    window_size = 120
    k_fold = 5
    data_inf = experiment_1_measure(window_size, k_fold, './ThisCarIsMineInf', config.feature_inf)
    data_lit = experiment_1_measure(window_size, k_fold, './ThisCarIsMineNormalized', config.feature_lit)
    with open(f'{path_to_save}/analyse_experiment_1.out_values.txt', 'w') as out:
        json.dump((data_lit, data_inf), out)

    with open(f'{path_to_save}/analyse_experiment_1.out_values.used-in-article.txt', 'r') as data_file:
        data_lit, data_inf = json.load(data_file)
        score_lit, roc_auc_lit, precision_lit, recall_lit = data_lit
        score_inf, roc_auc_inf, precision_inf, recall_inf = data_inf
        plot_experiment(score_lit, score_inf, fig_name=experiment_name, path=path_to_save)
        plot_roc_auc_dict(roc_auc_lit, roc_auc_inf, path=path_to_save)


def experiment_2():
    experiment_name = 'Experiment 2'
    path_to_save = './results/experiment_2'
    my_debug('Running', experiment_name, path=path_to_save)

    my_debug('Inf Theory', path=path_to_save)
    data_inf = experiment_2_measure('./ThisCarIsMineInf', config.feature_inf, path_to_save=path_to_save)
    my_debug('Literature', path=path_to_save)
    data_lit = experiment_2_measure('./ThisCarIsMineNormalized', config.feature_lit, path_to_save=path_to_save)
    with open(path_to_save + '/analyse_experiment_2.out_values.txt', 'w') as out:
        json.dump((data_lit, data_inf), out)

    with open(path_to_save + '/analyse_experiment_2.out_values.txt', 'r') as data_file:
        data_lit, data_inf = json.load(data_file)
        score_lit, roc_auc_lit = data_lit
        score_inf, roc_auc_inf = data_inf
        # Plot bars
        plot_experiment(score_lit, score_inf, ylim=(0, 1), fig_name=experiment_name, path=path_to_save)
        # Plot roc
        plot_roc_auc_dict(roc_auc_lit, roc_auc_inf, ylim=(0, 1), path=path_to_save)


def experiment_3():
    experiment_name = 'Experiment 3'
    my_debug('Running', experiment_name)
    path_to_save = './results/experiment_3'

    data_inf = experiment_3_measure('./ThisCarIsMineInf', config.feature_inf_120)
    data_lit = experiment_3_measure('./ThisCarIsMine', config.feature_lit)
    with open(path_to_save + '/analyse_experiment_3.out_values.txt', 'w') as out:
        json.dump((data_lit, data_inf), out)

    with open(path_to_save + '/analyse_experiment_3.out_values.txt', 'r') as data_file:
        data_lit, data_inf = json.load(data_file)
        score_lit, fit_time_lit, y_lit, y_pred_lit = data_lit
        score_inf, fit_time_inf, y_inf, y_pred_inf = data_inf
        # Plot bars
        plot_experiment(score_lit, score_inf, ylim=(0.0, .55), fig_name=experiment_name, path=path_to_save)
        # Plot
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_lit, y_pred_lit, cmap=plt.cm.Blues)
        disp.ax_.set_title('Confusion Matrix - Literature')
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_inf, y_pred_inf, cmap=plt.cm.Blues)
        disp.ax_.set_title('Confusion Matrix - Proposal')


def experiment_inf(experiment_name, feature_inf):
    if experiment_name not in ['hc', 'fs', 'hc_fs']:
        my_debug('Error: experiment_name invalid.')
        return
    my_debug('Running', experiment_name)
    path_to_save = f'./results/experiment/{experiment_name}'

    # window_size = 120
    # k_fold = 5
    # data_inf = experiment_1_measure(window_size, k_fold, './ThisCarIsMineInf', feature_inf)
    # data_lit = experiment_1_measure(window_size, k_fold, './ThisCarIsMineNormalized', config.feature_lit)
    # with open(f'{path_to_save}/analyse_{experiment_name}.out_values.txt', 'w') as out:
    #     json.dump((data_lit, data_inf), out)

    with open(f'{path_to_save}/analyse_{experiment_name}.out_values.txt', 'r') as data_file:
        data_lit, data_inf_hc, data_inf_fs, data_inf_hc_fs = json.load(data_file)
        score_lit, roc_auc_lit, precision_lit, recall_lit = data_lit
        score_inf_hc, roc_auc_inf_hc, precision_inf_hc, recall_inf_hc = data_inf_hc
        # score_inf_fs, roc_auc_inf_fs, precision_inf_fs, recall_inf_fs = data_inf_fs
        # score_inf_hc_fs, roc_auc_inf_hc_fs, precision_inf_hc_fs, recall_inf_hc_fs = data_inf_hc_fs
        #
        plot_experiment(score_lit, score_inf_hc, score_name='Accuracy', fig_name=experiment_name + '-accuracy', path=path_to_save)
        plot_roc_auc_dict(roc_auc_lit, roc_auc_inf_hc, path=path_to_save)
        plot_experiment(precision_lit, precision_inf_hc, score_name='Precision', fig_name=experiment_name + '-precison', path=path_to_save)
        plot_experiment(recall_lit, recall_inf_hc, score_name='Recall', fig_name=experiment_name + '-recall', path=path_to_save)

    # with open(f'{path_to_save}/analyse_{experiment_name}.out_values.txt', 'r') as data_file:
    #     data_lit, data_inf = json.load(data_file)
    #     score_lit, roc_auc_lit, precision_lit, recall_lit = data_lit
    #     score_inf, roc_auc_inf, precision_inf, recall_inf = data_inf
    #     # plot_experiment(score_lit, score_inf, fig_name=experiment_name, path=path_to_save)
    #     # plot_roc_auc_dict(roc_auc_lit, roc_auc_inf, path=path_to_save)
    #     plot_experiment(precision_lit, precision_inf, fig_name=experiment_name + '-precison', path=path_to_save)
    #     plot_experiment(recall_lit, recall_inf, fig_name=experiment_name + '-recall', path=path_to_save)


def experiment_inf_2(experiment_name, feature_inf_hc, feature_inf_fs, feature_inf_hc_fs):
    my_debug('Running', experiment_name)
    path_to_save = f'./results/experiment/'

    # window_size = 120
    # k_fold = 5
    # dataset_path = './ThisCarIsMineInf_window720_dx6'
    # data_inf_hc = experiment_1_measure(window_size, k_fold, dataset_path, feature_inf_hc)
    # data_inf_fs = experiment_1_measure(window_size, k_fold, dataset_path, feature_inf_fs)
    # data_inf_hc_fs = experiment_1_measure(window_size, k_fold, dataset_path, feature_inf_hc_fs)
    # data_lit = experiment_1_measure(window_size, k_fold, './ThisCarIsMineNormalized', config.feature_lit_remaining)
    # with open(f'{path_to_save}/analyse_{experiment_name}.out_values.txt', 'w') as out:
    #     json.dump((data_lit, data_inf_hc, data_inf_fs, data_inf_hc_fs), out)

    with open(f'{path_to_save}/analyse_{experiment_name}.out_values.txt', 'r') as data_file:
        data_lit, data_inf_hc, data_inf_fs, data_inf_hc_fs = json.load(data_file)
        score_lit, roc_auc_lit, precision_lit, recall_lit = data_lit
        score_inf_hc, roc_auc_inf_hc, precision_inf_hc, recall_inf_hc = data_inf_hc
        score_inf_fs, roc_auc_inf_fs, precision_inf_fs, recall_inf_fs = data_inf_fs
        score_inf_hc_fs, roc_auc_inf_hc_fs, precision_inf_hc_fs, recall_inf_hc_fs = data_inf_hc_fs
        #
        plot_experiment_4bars(score_lit, score_inf_hc, score_inf_fs, score_inf_hc_fs, 'Accuracy',
                              fig_name=experiment_name, path=path_to_save)
        plot_roc_auc_dict_4bars(roc_auc_lit, roc_auc_inf_hc, roc_auc_inf_fs, roc_auc_inf_hc_fs,
                                path=path_to_save)
        plot_experiment_4bars(precision_lit, precision_inf_hc, precision_inf_fs, precision_inf_hc_fs, 'Precision',
                              fig_name=experiment_name, path=path_to_save)
        plot_experiment_4bars(recall_lit, recall_inf_hc, recall_inf_fs, recall_inf_hc_fs, 'Recall',
                              fig_name=experiment_name, path=path_to_save)


def process_file(path, driver, num_files):
    df = []
    for i in range(1, 1 + 1, 1):
        filein = f'{path}/{driver}/All_{i}.csv'
        df.append(pd.read_csv(filein)[config.feature_inf_hcfs])
    return pd.concat(df)


if __name__ == '__main__':
    # experiment_1()
    # experiment_2()
    # experiment_3()

    # my_debug('\n#################################################################')
    # my_debug('### Experiment with HC')
    # experiment_inf(experiment_name='hc', feature_inf=config.feature_inf_hc)
    # my_debug('\n#################################################################')
    # my_debug('### Experiment with FS')
    # experiment_inf(experiment_name='fs', feature_inf=config.feature_inf_fs)
    # my_debug('\n#################################################################')
    # my_debug('### Experiment with HC and FS')
    # experiment_inf(experiment_name='hc_fs', feature_inf=config.feature_inf_hcfs)

    # experiment_inf(experiment_name='hc_fs', feature_inf=config.feature_inf_hc)
    experiment_inf_2(
        experiment_name='inf',
        # feature_inf_hc=config.feature_inf_hc,
        # feature_inf_fs=config.feature_inf_fs,
        # feature_inf_hc_fs=config.feature_inf_hcfs
        feature_inf_hc=config.feature_inf_remaining_hc,
        feature_inf_fs=config.feature_inf_remaining_fs,
        feature_inf_hc_fs=config.feature_inf_remaining
    )

    # experiment_sample_size()
    # plt.show()

    # path = './ThisCarIsMineInf'
    # dfa = process_file(path, 'A', 8)
    # dfb = process_file(path, 'B', 8)
    # dfc = process_file(path, 'C', 5)
    # dfd = process_file(path, 'D', 9)
    #
    # my_debug(f'\ndfa {dfa.shape[0]}, {dfa.dropna().shape[0]}, {dfa.dropna().shape[0]/dfa.shape[0]}\n',
    #       dfa.isna().sum() / dfa.shape[0])
    # my_debug(f'\ndfb {dfb.shape[0]}\n', dfb.isna().sum() / dfb.shape[0])
    # my_debug(f'\ndfc {dfc.shape[0]}\n', dfc.isna().sum() / dfc.shape[0])
    # my_debug(f'\ndfd {dfd.shape[0]}\n', dfd.isna().sum() / dfd.shape[0])
    #
    # df = pd.concat((dfa, dfb, dfc, dfd))
    # feat_nan = df.columns[df.isna().any()].tolist()
    # feat_without_nan = df.drop(feat_nan, axis=1).columns
    # my_debug(f'\ndf')
    # my_debug('With NaN: ', len(feat_nan), '\n', feat_nan)
    # my_debug('Without NaN', len(feat_without_nan), '\n', feat_without_nan)
