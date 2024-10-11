#!/usr/bin/python3

# Bibliotecas padrão
import os
import json
import math
import warnings
import multiprocessing

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Configurações do projeto
import config


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


def plot_experiment_4bars(score_lit, score_inf_hc, score_inf_fs, score_inf_hcfs, score_name, ylim=(0.5, 1.05),
                          fig_name='', path=None):
    font_size = 24
    bottom_size = 0.4
    X_axis = np.arange(len(classifier_names))
    bar_width = 0.2

    plt.figure(f'{score_name} Score. {fig_name}', figsize=config.default_figsize)
    x_lit_score = [np.mean(value) for value in score_lit.values()]
    x_inf_hc_score = [np.mean(value) for value in score_inf_hc.values()]
    x_inf_fs_score = [np.mean(value) for value in score_inf_fs.values()]
    x_inf_hc_fs_score = [np.mean(value) for value in score_inf_hcfs.values()]
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
            label='Proposal HC', color=color2, edgecolor="black")
    plt.bar(X_axis + 0.1, x_inf_hc_score, bar_width,
            yerr=[sem(value) for value in score_inf_fs.values()],
            label='Proposal FS', color=color3, edgecolor="black")
    plt.bar(X_axis + 0.3, x_inf_hc_fs_score, bar_width,
            yerr=[sem(value) for value in score_inf_hcfs.values()],
            label='Proposal HC+FS', color=color4, edgecolor="black")
    plt.ylim(ylim)
    plt.ylabel(f'{score_name} Score', fontsize=font_size)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)
    if path is not None:
        plt.savefig(f'{path}/{score_name}_score_results.png')


def plot_roc_auc_dict_4bars(roc_auc_lit, roc_auc_inf_hc, roc_auc_inf_fs, roc_auc_inf_hc_fs, ylim=(0.5, 1.05),
                            path=None):
    font_size = 24
    bottom_size = 0.4
    X_axis = np.arange(len(classifier_names))
    bar_width = 0.2

    plt.figure('ROC AUC', figsize=config.default_figsize)
    x_lit = [np.mean(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    lit_err = [sem(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    x_inf_hc = [np.mean(roc_auc_inf_hc[clf_name]) for clf_name in classifier_names]
    inf_err_hc = [sem(roc_auc_inf_hc[clf_name]) for clf_name in classifier_names]
    x_inf_fs = [np.mean(roc_auc_inf_fs[clf_name]) for clf_name in classifier_names]
    inf_err_fs = [sem(roc_auc_inf_fs[clf_name]) for clf_name in classifier_names]
    x_inf_hc_fs = [np.mean(roc_auc_inf_hc_fs[clf_name]) for clf_name in classifier_names]
    inf_err_hc_fs = [sem(roc_auc_inf_hc_fs[clf_name]) for clf_name in classifier_names]
    my_debug('ROC AUC, ')
    my_debug('Literature', min(x_lit), max(x_lit))
    my_debug('Proposal HC', min(x_inf_hc), max(x_inf_hc))
    my_debug('Proposal FS', min(x_inf_fs), max(x_inf_fs))
    my_debug('Proposal HC + FS', min(x_inf_hc_fs), max(x_inf_hc_fs))
    plt.bar(X_axis - 0.3, x_lit, bar_width, yerr=lit_err, label='Literature', color=color1, edgecolor="black")
    plt.bar(X_axis - 0.1, x_inf_hc, bar_width, yerr=inf_err_hc, label='HC', color=color2, edgecolor="black")
    plt.bar(X_axis + 0.1, x_inf_fs, bar_width, yerr=inf_err_fs, label='FS', color=color3, edgecolor="black")
    plt.bar(X_axis + 0.3, x_inf_hc_fs, bar_width, yerr=inf_err_hc_fs, label='HC+FS', color=color4, edgecolor="black")
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


def experiment_measure(window_size, k_fold, path, feature):
    filter_shannon = []
    for f in feature:
        if 'shannon' not in f:
            filter_shannon.append(f)
    feature = filter_shannon
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


def experiment_controler(experiment_name, feature_inf_hc, feature_inf_fs, feature_inf_hc_fs):
    my_debug('Running', experiment_name)
    directory_to_save = f'./results/experiment'

    window_size = 120
    k_fold = 5
    dataset_path = './datasets/ThisCarIsMineInf_window720_dx6'
    data_inf_hc = experiment_measure(window_size, k_fold, dataset_path, feature_inf_hc)
    data_inf_fs = experiment_measure(window_size, k_fold, dataset_path, feature_inf_fs)
    data_inf_hc_fs = experiment_measure(window_size, k_fold, dataset_path, feature_inf_hc_fs)
    data_lit = experiment_measure(window_size, k_fold, './datasets/ThisCarIsMineNormalized', config.feature_lit_remaining)
    if not os.path.exists(f'{directory_to_save}/analyse_{experiment_name}.out_values.txt'):
        os.makedirs(directory_to_save, exist_ok=True)
    with open(f'{directory_to_save}/analyse_{experiment_name}.out_values.txt', 'w') as out:
        json.dump((data_lit, data_inf_hc, data_inf_fs, data_inf_hc_fs), out)

    with open(f'{directory_to_save}/analyse_{experiment_name}.out_values.txt', 'r') as data_file:
        data_lit, data_inf_hc, data_inf_fs, data_inf_hc_fs = json.load(data_file)
        score_lit, roc_auc_lit, precision_lit, recall_lit = data_lit
        score_inf_hc, roc_auc_inf_hc, precision_inf_hc, recall_inf_hc = data_inf_hc
        score_inf_fs, roc_auc_inf_fs, precision_inf_fs, recall_inf_fs = data_inf_fs
        score_inf_hc_fs, roc_auc_inf_hc_fs, precision_inf_hc_fs, recall_inf_hc_fs = data_inf_hc_fs
        #
        plot_experiment_4bars(score_lit, score_inf_hc, score_inf_fs, score_inf_hc_fs, 'Accuracy',
                              fig_name=experiment_name, path=directory_to_save)
        plot_roc_auc_dict_4bars(roc_auc_lit, roc_auc_inf_hc, roc_auc_inf_fs, roc_auc_inf_hc_fs,
                                path=directory_to_save)
        plot_experiment_4bars(precision_lit, precision_inf_hc, precision_inf_fs, precision_inf_hc_fs, 'Precision',
                              fig_name=experiment_name, path=directory_to_save)
        plot_experiment_4bars(recall_lit, recall_inf_hc, recall_inf_fs, recall_inf_hc_fs, 'Recall',
                              fig_name=experiment_name, path=directory_to_save)


if __name__ == '__main__':
    experiment_controler(
        experiment_name='inf',
        feature_inf_hc=config.feature_inf_hc,
        feature_inf_fs=config.feature_inf_fs,
        feature_inf_hc_fs=config.feature_inf_hcfs
    )
