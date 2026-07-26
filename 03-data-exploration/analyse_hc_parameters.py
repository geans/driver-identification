import json
import logging
import multiprocessing
import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.utils.multiclass import type_of_target

import config

warnings.filterwarnings("ignore")

color1 = '#115f9a'
color2 = '#009dc3'
color3 = '#00d38d'
color4 = '#d0f400'

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyse_hc_parameters(directory_input, directory_output):
    X_names = []
    path_input = []
    for series_length in range(60, 901, 60):
        _path = f"{directory_input}/{series_length}/ThisCarIsMine"
        X_names.append(str(series_length))
        path_input.append(_path)

    def classifier_handle(accuracy, roc_auc, precision, recall):
        try:
            # logger.info(f'y.dtype: {y.dtype}')
            # logger.info(f'y.iloc[0] type: {type(y.iloc[0])}')
            # logger.info(f'y.unique(): {y.unique()}')
            # logger.info(f"type_of_target: {type_of_target(y)}")

            # exit()
            # To predict
            y_pred = cross_val_predict(clf, X, y, cv=5)

            # Get scores
            accuracy_value = metrics.accuracy_score(y, y_pred)
            fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
            roc_auc_value = metrics.auc(fpr, tpr)
            precision_value = metrics.precision_score(y, y_pred)
            recall_value = metrics.recall_score(y, y_pred)

            # Add scores to external list
            accuracy.append(accuracy_value)
            roc_auc.append(roc_auc_value)
            precision.append(precision_value)
            recall.append(recall_value)
        except Exception as e:
            logger.error(f'Error in "analyse_hc_parameters::classifier_handle". (clf, X, y) = ({(clf, X, y)}). {e}', exc_info=True)
            exit()

    accuracy_list = []
    accuracy_sem_list = []

    roc_auc_list = []
    roc_auc_sem_list = []

    precision_list = []
    precision_sem_list = []

    recall_list = []
    recall_sem_list = []

    # classifier = SVC
    classifier = RandomForestClassifier
    print(classifier)
    for path in path_input:
        df_list = []
        for driver in 'ABCD':
            _df = pd.read_csv(path + f'/{driver}/All_1.csv')[config.feature_inf_remaining].dropna() # TODO: refazer caminho para arquivo
            _df['driver'] = [driver] * _df.shape[0]
            df_list.append(_df)
        df = pd.concat(df_list)
        print(path, df.shape)
        X = df.drop(['driver'], axis=1)
        manager = multiprocessing.Manager()
        accuracy = manager.list()
        roc_auc = manager.list()
        precision = manager.list()
        recall = manager.list()
        running_process = []
        for driver in 'ABCD':
            # y = df['driver'].replace(['A', 'B', 'C', 'D'],
            #                          ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
            y = (df['driver'] == driver)
            clf = classifier()
            p = multiprocessing.Process(target=classifier_handle,
                                        args=(accuracy, roc_auc, precision, recall))
            p.start()
            running_process.append(p)
        for p in running_process:
            p.join()

        accuracy_list.append(np.mean(accuracy))
        accuracy_sem_list.append(sem(accuracy))

        roc_auc_list.append(np.mean(roc_auc))
        roc_auc_sem_list.append(sem(roc_auc))

        precision_list.append(np.mean(precision))
        precision_sem_list.append(sem(precision))

        recall_list.append(np.mean(recall))
        recall_sem_list.append(sem(recall))


    if not os.path.exists(directory_output):
        os.makedirs(directory_output, exist_ok=True)
    with open(directory_output + '/analyse_data__analyse_hc_parameters.log', 'w') as out:
        json.dump(
            (
                accuracy_list, accuracy_sem_list,
                roc_auc_list, roc_auc_sem_list,
                precision_list, precision_sem_list,
                recall_list, recall_sem_list,
            ),
            out
        )
    with open(directory_output + '/analyse_data__analyse_hc_parameters.log', 'r') as data_file:
        accuracy_list, accuracy_sem_list, \
            roc_auc_list, roc_auc_sem_list, \
            precision_list, precision_sem_list, \
            recall_list, recall_sem_list = json.load(data_file)
        # print(len(accuracy_list), len(accuracy_sem_list))
        # print(len(roc_auc_list), len(roc_auc_sem_list))
        # print(len(precision_list), len(precision_sem_list))
        # print(len(recall_list), len(recall_sem_list))
        # print()
        logger.info(f'Precision: {precision_list}, SEM: {precision_sem_list}')
        logger.info(f'Recall: {recall_list}, SEM: {recall_sem_list}')
        
        font_size = 24
        legend_font_size = 18
        bar_width = 0.2
        x, y = config.default_figsize
        plt.figure(figsize=(x * 2, y))
        X_axis = np.arange(len(X_names))
        plt.bar(X_axis - 0.3, accuracy_list, bar_width,
                yerr=accuracy_sem_list, label='Accuracy', color=color1, edgecolor="black")
        plt.bar(X_axis - 0.1, roc_auc_list, bar_width,
                yerr=roc_auc_sem_list, label='ROC AUC', color=color2, edgecolor="black")
        plt.bar(X_axis + 0.1, precision_list, bar_width,
                yerr=precision_sem_list, label='Precision', color=color3, edgecolor="black")
        plt.bar(X_axis + 0.3, recall_list, bar_width,
                yerr=recall_sem_list, label='Recall', color=color4, edgecolor="black")
        plt.xticks(X_axis, X_names)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        # plt.tick_params(axis='x', labelrotation=90)
        plt.subplots_adjust(bottom=.15)
        plt.ylim((0.5, 1))
        plt.xlabel('Length', fontsize=font_size)
        plt.ylabel('Score', fontsize=font_size)
        plt.legend(fontsize=font_size - 4)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
                   fancybox=True, shadow=True, ncol=2, fontsize=legend_font_size)

        major_ticks = np.arange(.5, 1.01, .1)
        minor_ticks = np.arange(.5, 1.01, .05)
        plt.yticks(major_ticks)
        plt.yticks(minor_ticks, minor=True)
        plt.grid(which='major', alpha=.5)
        plt.grid(which='minor', alpha=.2)
        plt.grid(axis='x')

        plt.savefig(directory_output + '/analyse_data__analyse_hc_parameters.png')


if __name__ == '__main__':
    analyse_hc_parameters(
        directory_input='../02-transformation/02.1-dataset-processed/multiprocessing',
        directory_output='results/analyse_hc_parameters'
    )