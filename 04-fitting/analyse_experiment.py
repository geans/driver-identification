# Bibliotecas padrão
import json
import logging
import math
import multiprocessing
import os
import time
import warnings

# Configurações do projeto
import config
import matplotlib.pyplot as plt

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#from deeplearningclassifier import LSTMClassifier

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


def my_debug(*objects, sep=' ', end='\n', file=None, flush=False, path='.'):
    if config.debug_on_screen:
        print(*objects, sep=sep, end=end, file=file, flush=flush)
    # output_file = open(f'{path}/log.txt', 'a')
    # for obj in objects:
    #     output_file.write(str(obj))
    #     output_file.write(sep)
    # output_file.write(end)
    # output_file.close()


classifier_names = (
    "kNN",
    "Linear SVM",
    "RBF SVM",
    "D. Tree",
    "R. Forest",
    "MLP",
    "N. Bayes",
    # "LSTM"
)

classifiers = (
    KNeighborsClassifier(math.floor(math.sqrt(config.inf_window_size))),
    SVC(kernel="linear"),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    GaussianNB(),
    # LSTMClassifier()
)


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
    plt.bar(
        X_axis - 0.1, 
        x_lit_score, 
        bar_width,
        yerr=[sem(value) for value in score_lit.values()],
        label='Literature', 
        color=color1, 
        edgecolor="black"
    )
    plt.bar(
        X_axis + 0.1,
        x_inf_hc_fs_score, 
        bar_width,
        yerr=[sem(value) for value in score_inf_hcfs.values()],
        label='Proposal', 
        color=color4, 
        edgecolor="black"
    )
    plt.ylim(ylim)
    plt.ylabel(f'{score_name} Score', fontsize=font_size)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)
    if path is not None:
        plt.savefig(f'{path}/{score_name}_score_results.svg')


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
    plt.bar(
        X_axis - 0.1, 
        x_lit, 
        bar_width, 
        yerr=lit_err, 
        label='Literature', 
        color=color1, 
        edgecolor="black"
    )
    plt.bar(
        X_axis + 0.1, 
        x_inf_hc_fs,
        bar_width, 
        yerr=inf_err_hc_fs, 
        label='Proposal', 
        color=color4, 
        edgecolor="black"
    )
    plt.ylim(ylim)
    plt.ylabel('ROC AUC', fontsize=font_size)
    plt.xticks(X_axis, classifier_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -bottom_size),
               fancybox=True, shadow=True, ncol=3, fontsize=font_size)
    plt.subplots_adjust(bottom=bottom_size)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(axis='x', labelrotation=45)
    if path is not None:
        plt.savefig(f'{path}/roc_auc_mean.svg')


def combined_time_series_split(groups, n_splits):
    """
    Aplica TimeSeriesSplit dentro de cada grupo (driver) e combina
    os índices de treino/teste de todos os grupos, fold a fold.
    Garante que, para cada driver, o teste é sempre posterior ao treino.
    Também garante ordem dos dados (temporal).
    """
    unique_groups = pd.unique(groups)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    group_positions = {g: np.where(groups == g)[0] for g in unique_groups}
    group_splits = {g: list(tscv.split(pos)) for g, pos in group_positions.items()}

    for fold in range(n_splits):
        train_idx, test_idx = [], []
        for g, positions in group_positions.items():
            tr, te = group_splits[g][fold]
            train_idx.extend(positions[tr])
            test_idx.extend(positions[te])
        yield np.array(train_idx), np.array(test_idx)


def experiment_measure(window_size, k_fold, path, feature):
    def classifier_handle(args_dict):
        score_dict = args_dict['score_dict']
        roc_auc_dict = args_dict['roc_auc_dict']
        precision_dict = args_dict['precision_dict']
        recall_dict = args_dict['recall_dict']
        train_time_dict = args_dict['train_time_dict']
        pred_time_dict = args_dict['pred_time_dict']
        clf = args_dict['clf']
        clf_name = args_dict['clf_name']
        X = args_dict['X']
        y = args_dict['y']
        k_fold = args_dict['k_fold']
        groups = args_dict['groups']  # np.array com o driver de origem de cada linha

        y_pred = np.full(len(y), False)
        tested_mask = np.zeros(len(y), dtype=bool)  # nem toda linha é testada em TSS

        for train_idx, test_idx in combined_time_series_split(groups, k_fold):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]

            t0 = time.time()
            clf.fit(X_train, y_train)
            tf = time.time()
            train_time_dict[clf_name] = train_time_dict.get(clf_name, []) + [tf - t0]

            t0 = time.time()
            y_pred[test_idx] = clf.predict(X_test)
            tf = time.time()
            pred_time_dict[clf_name] = pred_time_dict.get(clf_name, []) + [tf - t0]
            tested_mask[test_idx] = True

        # Métricas só sobre as linhas que de fato foram testadas
        y_eval, y_pred_eval = y[tested_mask], y_pred[tested_mask]
        fpr, tpr, threshold = metrics.roc_curve(y_eval, y_pred_eval)
        roc_auc_dict[clf_name] = roc_auc_dict.get(clf_name, []) + [metrics.auc(fpr, tpr)]
        score_dict[clf_name] = score_dict.get(clf_name, []) + [metrics.accuracy_score(y_eval, y_pred_eval)]
        precision_dict[clf_name] = precision_dict.get(clf_name, []) + [metrics.precision_score(y_eval, y_pred_eval)]
        recall_dict[clf_name] = recall_dict.get(clf_name, []) + [metrics.recall_score(y_eval, y_pred_eval)]

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
        )):
            trip = trips[t]
            csv_filename = f'{path}/{driver}/All_{trip}.csv'
            _df = pd.read_csv(csv_filename, index_col=False)

            df_arr += [_df[feature].dropna()]
            my_debug(csv_filename, df_arr[-1].shape)
            limit = min(limit, df_arr[-1].shape[0])
        # limit = 150  # Utilizado para teste rápido, comentar para rodar o experimento completo
        my_debug('limit =', limit)
        for i, driver in enumerate('ABCD'):
            df_arr[i] = df_arr[i][:limit]
            df_arr[i][class_feat] = [driver] * df_arr[i].shape[0]
        df = pd.concat(df_arr)
        total_process = len(classifier_names) * len('ABCD')
        my_debug('total_process =', total_process)
        counter = 0
        # Process each window
        for window in [df]:
            if len(window) < window_size:
                my_debug('len(window)', len(window), 'window_size', window_size)
                continue
            X = window.drop([class_feat], axis=1)
            groups = window['driver'].values
            for clf, clf_name in zip(classifiers, classifier_names):
                running_process = []
                for driver in 'ABCD':
                    y = window[class_feat].eq(driver)
                    p = multiprocessing.Process(
                        target=classifier_handle,
                        args=(
                            {
                                "score_dict": score_dict,
                                "roc_auc_dict": roc_auc_dict,
                                "precision_dict": precision_dict,
                                "recall_dict": recall_dict,
                                "train_time_dict": train_time_dict,
                                "pred_time_dict": pred_time_dict,
                                "clf": clf,
                                "clf_name": clf_name,
                                "X": X,
                                "y": y,
                                "k_fold": k_fold,
                                "groups": groups,
                            },
                        )
                    )
                    p.start()
                    running_process.append(p)
                for p in running_process:
                    p.join()
                    counter += 1
                    my_debug(f'{t} | {counter}/{total_process}', end='\r')
        my_debug()
    return score_dict.copy(), roc_auc_dict.copy(), precision_dict.copy(), recall_dict.copy()


def experiment_controler(
        experiment_name, 
        feature_inf_hc_fs,
        dataset_path
    ):
    my_debug('Running', experiment_name)
    directory_to_save = './results/experiment'

    window_size = 120
    k_fold = 10
    data_inf_hc_fs = experiment_measure(window_size, k_fold, dataset_path, feature_inf_hc_fs)
    data_lit = experiment_measure(window_size, k_fold, '../02-transformation/02.1-dataset-processed/ThisCarIsMineNormalized', config.feature_lit_remaining)
    if not os.path.exists(directory_to_save):
        os.makedirs(directory_to_save, exist_ok=True)
    with open(f'{directory_to_save}/analyse_{experiment_name}.out_values.txt', 'w') as out:
        json.dump((data_lit, data_inf_hc_fs), out)

    with open(f'{directory_to_save}/analyse_{experiment_name}.out_values.txt', 'r') as data_file:
        data_lit, data_inf_hc_fs = json.load(data_file)
        score_lit, roc_auc_lit, precision_lit, recall_lit = data_lit
        score_inf_hc_fs, roc_auc_inf_hc_fs, precision_inf_hc_fs, recall_inf_hc_fs = data_inf_hc_fs
        
        plot_experiment_4bars(score_lit, score_inf_hc_fs, 'Accuracy',
                              fig_name=experiment_name, path=directory_to_save)
        plot_roc_auc_dict_4bars(roc_auc_lit, roc_auc_inf_hc_fs,
                                path=directory_to_save)
        plot_experiment_4bars(precision_lit, precision_inf_hc_fs, 'Precision',
                              fig_name=experiment_name, path=directory_to_save)
        plot_experiment_4bars(recall_lit, recall_inf_hc_fs, 'Recall',
                              fig_name=experiment_name, path=directory_to_save)


if __name__ == '__main__':
    for dataset_path, experiment_name, configuration in (
        (
            '../02-transformation/02.1-dataset-processed/multiprocessing/720/ThisCarIsMine',
            'information',
            config.feature_inf_hcfs
        ),
        # (
        #     '../02-transformation/02.1-dataset-processed/ThisCarIsMineNormalized', 
        #     'literature',
        #     config.feature_lit_remaining
        # )
    ):
        if not os.path.exists(dataset_path):
            logger.error(f'Dataset not found in {dataset_path}, aborting experiment.')
        else:
            experiment_controler(
                experiment_name=experiment_name,
                feature_inf_hc_fs=configuration,
                dataset_path=dataset_path
            )
