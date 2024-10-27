#!/usr/bin/python3

# Bibliotecas padrão
import os
import json
import math
import time
import warnings
import multiprocessing

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Configurações do projeto
import config
from deeplearningclassifier import LSTMClassifier


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
    # "kNN",
    # "Linear SVM",
    # "RBF SVM",
    # "D. Tree",
    # "R. Forest",
    # "MLP",
    # "N. Bayes",
    "LSTM"
]

classifiers = [
    # KNeighborsClassifier(math.floor(math.sqrt(config.inf_window_size))),
    # SVC(kernel="linear"),
    # SVC(),
    # DecisionTreeClassifier(),
    # RandomForestClassifier(),
    # MLPClassifier(),
    # GaussianNB(),
    LSTMClassifier()
]


def plot_experiment_4bars(score_lit, score_inf_hcfs, score_name, ylim=(0.5, 1.05),
                          fig_name='', path=None):
    font_size = 24
    bottom_size = 0.4
    X_axis = np.arange(len(classifier_names))
    bar_width = 0.2

    plt.figure(f'{score_name} Score. {fig_name}', figsize=config.default_figsize)
    x_lit_score = [np.mean(value) for value in score_lit.values()]
    x_inf_hc_fs_score = [np.mean(value) for value in score_inf_hcfs.values()]
    my_debug(f'{score_name}, length:', len(list(score_lit.values())[0]))
    my_debug('Literature', min(x_lit_score), max(x_lit_score))
    my_debug('Proposal', min(x_inf_hc_fs_score), max(x_inf_hc_fs_score))
    plt.bar(X_axis - 0.3, x_lit_score, bar_width,
            yerr=[sem(value) for value in score_lit.values()],
            label='Literature', color=color1, edgecolor="black")
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


def plot_roc_auc_dict_4bars(roc_auc_lit, roc_auc_inf_hc_fs, ylim=(0.5, 1.05),
                            path=None):
    font_size = 24
    bottom_size = 0.4
    X_axis = np.arange(len(classifier_names))
    bar_width = 0.2

    plt.figure('ROC AUC', figsize=config.default_figsize)
    x_lit = [np.mean(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    lit_err = [sem(roc_auc_lit[clf_name]) for clf_name in classifier_names]
    x_inf_hc_fs = [np.mean(roc_auc_inf_hc_fs[clf_name]) for clf_name in classifier_names]
    inf_err_hc_fs = [sem(roc_auc_inf_hc_fs[clf_name]) for clf_name in classifier_names]
    my_debug('ROC AUC, ')
    my_debug('Literature', min(x_lit), max(x_lit))
    my_debug('Proposal', min(x_inf_hc_fs), max(x_inf_hc_fs))
    plt.bar(X_axis - 0.3, x_lit, bar_width, yerr=lit_err, label='Literature', color=color1, edgecolor="black")
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
    # def classifier_handle(score_dict, roc_auc_dict, precision_dict, recall_dict,
    #                       clf, clf_name, X, y, k_fold):
    #     t0 = time.time()
    #     y_pred = cross_val_predict(clf, X, y, cv=k_fold)
    #     tf = time.time()
    #     # train_time_dict[clf_name] = train_time_dict.get(clf_name, []) + [ (tf - t0) / k_fold ]
    #     fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
    #     roc_auc_dict[clf_name] = roc_auc_dict.get(clf_name, []) + [metrics.auc(fpr, tpr)]
    #     score_dict[clf_name] = score_dict.get(clf_name, []) + [metrics.accuracy_score(y, y_pred)]
    #     precision_dict[clf_name] = precision_dict.get(clf_name, []) + [metrics.precision_score(y, y_pred)]
    #     recall_dict[clf_name] = recall_dict.get(clf_name, []) + [metrics.recall_score(y, y_pred)]
    def classifier_handle(score_dict, roc_auc_dict, precision_dict, recall_dict, train_time_dict, pred_time_dict, 
                          clf, clf_name, X, y, k_fold):
        # Criar a divisão dos folds usando KFold
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        
        for train_index, test_index in kf.split(X):
            # Separar os dados de treino e teste para o fold atual
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train the classifier
            t0 = time.time()
            clf.fit(X_train, y_train)
            tf = time.time()
            train_time_dict[clf_name] = train_time_dict.get(clf_name, []) + [t0 - tf]
            
            # Predict on test set
            t0 = time.time()
            y_pred = clf.predict(X_test)
            tf = time.time()
            pred_time_dict[clf_name] = pred_time_dict.get(clf_name, []) + [t0 - tf]

            # Evalue measures
            fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
            roc_auc_dict[clf_name] = roc_auc_dict.get(clf_name, []) + [metrics.auc(fpr, tpr)]
            score_dict[clf_name] = score_dict.get(clf_name, []) + [metrics.accuracy_score(y_test, y_pred)]
            precision_dict[clf_name] = precision_dict.get(clf_name, []) + [metrics.precision_score(y_test, y_pred)]
            recall_dict[clf_name] = recall_dict.get(clf_name, []) + [metrics.recall_score(y_test, y_pred)]

    class_feat = 'driver'
    manager = multiprocessing.Manager()

    score_dict = manager.dict()
    roc_auc_dict = manager.dict()
    precision_dict = manager.dict()
    recall_dict = manager.dict()
    train_time_dict = manager.dict()
    pred_time_dict = manager.dict()
    df_arr = []
    for t in range(1):
        limit = 99999
        for driver, trips in zip('ABCD', (
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                # [1, 2, 3, 4, 5],
                # [1, 2, 3, 4, 6],
                # [1, 2, 3, 4, 5],
                # [1, 3, 5, 6, 7],
        )):
            trip = trips[t]
            csv_filename = f'{path}/{driver}/All_{trip}.csv'
            _df = pd.read_csv(csv_filename, index_col=False)

            df_arr += [_df[feature].dropna()]
            my_debug(csv_filename, df_arr[-1].shape)
            limit = min(limit, df_arr[-1].shape[0])
        limit = 150
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
                                                args=(score_dict, roc_auc_dict, precision_dict, recall_dict, train_time_dict, pred_time_dict,
                                                      clf, clf_name, X, y, k_fold))
                    p.start()
                    running_process.append(p)
                for p in running_process:
                    p.join()
                    counter += 1
                    my_debug(f'{t} | {counter}/{total_process}', end='\r')
        my_debug()
    return score_dict.copy(), roc_auc_dict.copy(), precision_dict.copy(), recall_dict.copy()


def experiment_controler(experiment_name, feature_inf_hc_fs):
    my_debug('Running', experiment_name)
    directory_to_save = f'./results/experiment'

    window_size = 120
    k_fold = 5
    dataset_path = './datasets/ThisCarIsMineInf_window720_dx6'
    data_inf_hc_fs = experiment_measure(window_size, k_fold, dataset_path, feature_inf_hc_fs)
    data_lit = experiment_measure(window_size, k_fold, './datasets/ThisCarIsMineNormalized', config.feature_lit_remaining)
    if not os.path.exists(directory_to_save):
        os.makedirs(directory_to_save, exist_ok=True)
    with open(f'{directory_to_save}/analyse_{experiment_name}.out_values.txt', 'w') as out:
        json.dump((data_lit, data_inf_hc_fs), out)

    with open(f'{directory_to_save}/analyse_{experiment_name}.out_values.txt', 'r') as data_file:
        data_lit, data_inf_hc_fs = json.load(data_file)
        score_lit, roc_auc_lit, precision_lit, recall_lit = data_lit
        score_inf_hc_fs, roc_auc_inf_hc_fs, precision_inf_hc_fs, recall_inf_hc_fs = data_inf_hc_fs
        #
        plot_experiment_4bars(score_lit, score_inf_hc_fs, 'Accuracy',
                              fig_name=experiment_name, path=directory_to_save)
        plot_roc_auc_dict_4bars(roc_auc_lit, roc_auc_inf_hc_fs,
                                path=directory_to_save)
        plot_experiment_4bars(precision_lit, precision_inf_hc_fs, 'Precision',
                              fig_name=experiment_name, path=directory_to_save)
        plot_experiment_4bars(recall_lit, recall_inf_hc_fs, 'Recall',
                              fig_name=experiment_name, path=directory_to_save)


if __name__ == '__main__':
    experiment_controler(
        experiment_name='inf',
        feature_inf_hc_fs=config.feature_inf_hcfs
    )
