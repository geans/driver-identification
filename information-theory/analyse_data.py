import json
import math
import multiprocessing
import time
from datetime import timedelta

import numpy
import ordpy
import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from numpy import mean
from ordpy import minimum_complexity_entropy, maximum_complexity_entropy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import sem
import seaborn as sns

import config
import data_preprocessing as dpp
from information import InformationHandle
from getdata import GetData

color1 = '#115f9a'
color2 = '#009dc3'
color3 = '#00d38d'
color4 = '#d0f400'


def getx(df):
    return df.drop([config.label, config.driver], axis=1)


def hc_limits(dx):
    m = 10
    hc_max = maximum_complexity_entropy(dx=dx, m=m)
    size = (math.factorial(dx) - 1) * m
    hc_min = minimum_complexity_entropy(dx=dx, size=size)
    #
    h_max = [x[0] for x in hc_max]
    c_max = [x[1] for x in hc_max]
    h_min = [x[0] for x in hc_min]
    c_min = [x[1] for x in hc_min]
    #
    return (h_min, c_min), (h_max, c_max)


def preprocessing_to_hc(series):
    series = series.loc[series.shift() != series]
    return series


def plot_limits(hc_min, hc_max):
    plt.plot(hc_min[0], hc_min[1], color="black", linewidth=0.8)
    plt.plot(hc_max[0], hc_max[1], color="black", linewidth=0.8)


def get_data_list(path='./ThisCarIsMine'):
    data_arr = [
        (pd.read_csv(path + '/A/All_1.csv'), 'A'),
        (pd.read_csv(path + '/A/All_2.csv'), 'A'),
        (pd.read_csv(path + '/A/All_3.csv'), 'A'),
        (pd.read_csv(path + '/A/All_4.csv'), 'A'),
        (pd.read_csv(path + '/A/All_5.csv'), 'A'),
        (pd.read_csv(path + '/A/All_6.csv'), 'A'),
        (pd.read_csv(path + '/A/All_7.csv'), 'A'),
        # (pd.read_csv(path + '/A/All_8.csv'), 'A'), # outlier
        #
        (pd.read_csv(path + '/B/All_1.csv'), 'B'),
        (pd.read_csv(path + '/B/All_2.csv'), 'B'),
        (pd.read_csv(path + '/B/All_3.csv'), 'B'),
        (pd.read_csv(path + '/B/All_4.csv'), 'B'),
        # (pd.read_csv(path + '/B/All_5.csv'), 'B'), # outlier
        (pd.read_csv(path + '/B/All_6.csv'), 'B'),
        (pd.read_csv(path + '/B/All_7.csv'), 'B'),
        (pd.read_csv(path + '/B/All_8.csv'), 'B'),
        #
        (pd.read_csv(path + '/C/All_1.csv'), 'C'),
        (pd.read_csv(path + '/C/All_2.csv'), 'C'),
        (pd.read_csv(path + '/C/All_3.csv'), 'C'),
        (pd.read_csv(path + '/C/All_4.csv'), 'C'),
        (pd.read_csv(path + '/C/All_5.csv'), 'C'),
        #
        (pd.read_csv(path + '/D/All_1.csv'), 'D'),
        (pd.read_csv(path + '/D/All_2.csv'), 'D'),
        (pd.read_csv(path + '/D/All_3.csv'), 'D'),
        (pd.read_csv(path + '/D/All_4.csv'), 'D'),
        (pd.read_csv(path + '/D/All_5.csv'), 'D'),
        (pd.read_csv(path + '/D/All_6.csv'), 'D'),
        (pd.read_csv(path + '/D/All_7.csv'), 'D'),
        (pd.read_csv(path + '/D/All_8.csv'), 'D'),
        (pd.read_csv(path + '/D/All_9.csv'), 'D'),
    ]
    return data_arr


def split_data_to_window(series, window_size, shift):
    pivot = 0
    sub_series_list = []
    len_list = len(series)
    while pivot + window_size <= len_list:
        sub_series_list.append(series[pivot:pivot + window_size])
        pivot += shift
    return sub_series_list


# def analyse_variance(dfx):
#     print('\n  # ANALYSE VARIANCE')
#     variance_df = dfx.loc[:, (dfx != dfx.iloc[0]).any()]
#     invariance = list(set(dfx.columns) - set(variance_df.columns))
#     print('feature_invariance =', invariance, '#', len(invariance))
#     print('feature_variance =', list(variance_df.columns), '#', variance_df.shape[1])
#     return invariance, variance_df


def get_first_trip(path=None):
    if path is None:
        path = './ThisCarIsMine'
    dfa = pd.read_csv(path + '/A/All_1.csv')
    dfb = pd.read_csv(path + '/B/All_1.csv')
    dfc = pd.read_csv(path + '/C/All_1.csv')
    dfd = pd.read_csv(path + '/D/All_1.csv')
    return pd.concat([dfa, dfb, dfc, dfd])


def analyse_indifferent(features=config.all_features):
    # data_arr = get_data_list()
    # df = pd.concat([i[0] for i in data_arr])
    df = get_first_trip()
    indifferent_features = [i for i in features if np.std(df[i]) == 0]
    return indifferent_features


def analyse_variance(features=config.all_features):
    print('\n  # VARIANCE')

    # data_arr = get_data_list()
    # dfa = pd.concat([i[0] for i in data_arr if i[1] == 'A'])
    # dfb = pd.concat([i[0] for i in data_arr if i[1] == 'B'])
    # dfc = pd.concat([i[0] for i in data_arr if i[1] == 'C'])
    # dfd = pd.concat([i[0] for i in data_arr if i[1] == 'D'])

    path = './ThisCarIsMine'
    dfa = pd.read_csv(path + '/A/All_1.csv')
    dfb = pd.read_csv(path + '/B/All_1.csv')
    dfc = pd.read_csv(path + '/C/All_1.csv')
    dfd = pd.read_csv(path + '/D/All_1.csv')

    a = [i for i in features if np.std(dfa[i]) == 0]
    b = [i for i in features if np.std(dfb[i]) == 0]
    c = [i for i in features if np.std(dfc[i]) == 0]
    d = [i for i in features if np.std(dfd[i]) == 0]
    # x = set(a + b + c + d)
    # print('Driver A:', len(a), a)
    # print('Driver B:', len(b), b)
    # print('Driver C:', len(c), c)
    # print('Driver D:', len(d), d)
    # print('Join drivers:', len(x), x)
    return list(set(a + b + c + d))


def analyse_correlation(df, correlate_threshold):
    print('\n  # ANALYSE CORRELATION')
    corr = df.corr()

    # write into file the min and max values correlations for each feature
    with open('correlation.txt', 'w') as c:
        for feature in corr:
            c.write(f';\n{feature}\n')
            sort_corr = corr[feature].dropna().sort_values()
            try:
                c.write(f'  {sort_corr[0]}\n')
                c.write(f'  {sort_corr[-2]}\n')
            except Exception as e:
                pass

    # read min and max value for each features to array to min: (mini) and to max (maxi). To print range
    with open('correlation.txt', 'r') as c:
        results = c.read().split(';')[1:]
        mini, maxi = [], []
        for r in results:
            try:
                feature, minor, major = r.split()
                mini.append(float(minor))
                maxi.append(float(major))
            except Exception as e:
                pass

    # print('Correlation')
    # print(f'  Negative range: [{min(mini)}, {max(mini)}]')
    # print(f'  Positive range: [{min(maxi)}, {max(maxi)}]')
    # print()

    # Filter by correlation
    included = []
    excluded = []
    columns = list(df.columns)
    for i in range(len(columns)):
        c1 = df[columns[i]]
        must_add = True
        for j in range(i + 1, len(columns), 1):
            c2 = df[columns[j]]
            if abs(c1.corr(c2)) > correlate_threshold:
                must_add = False
                break
        if must_add:
            included.append(columns[i])
        else:
            excluded.append(columns[i])
    # print('correlation =', correlate_threshold, '\n')
    # print('included =', included, '#', len(included), '\n')
    # print('excluded =', excluded, '#', len(excluded), '\n')

    # Calcular correlação entre todas as combinações de duas features
    correlation_matrix = corr

    # Nome do arquivo para salvar a correlação
    file_name = "correlation_output.txt"

    # Abrir o arquivo para escrita
    with open(file_name, 'w') as file:
        # Escrever os resultados de correlação no arquivo
        file.write("Correlation between features:\n")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]
                file.write(f"{feature1} - {feature2}: {correlation}\n")
    return corr, included, excluded


def plot_correlation(corr):
    print('\n  # PLOT CORRELATION')
    # f, ax = plt.subplots(figsize=(10, 8))
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # sns.set(font_scale=1.4)
    # sns.heatmap(corr,
    #             # cmap=sns.diverging_palette(220, 10, as_cmap=True),
    #             vmin=-1.0, vmax=1.0,
    #             square=True, ax=ax)  # , cmap='crest')
    # f.subplots_adjust(bottom=0.4)

    plt.matshow(corr, vmin=-1.0, vmax=1.0)
    cb = plt.colorbar()
    plt.savefig(f'correlation.png')

    # cb.ax.tick_params(labelsize=14)
    # plt.tick_params(labelsize=14)


def analyse_mutual_information():
    print('  # Mutual information')
    data_arr = get_data_list()
    df_list = []
    for d in data_arr:
        d[0]['class'] = d[0].shape[0] * [d[1]]
        df_list.append(d[0])
    df = pd.concat(df_list)
    X = df.drop(['class'], axis=1)
    y = df['class']
    mi = mutual_info_classif(X, y)
    mi = pd.Series(mi)
    mi.index = X.columns
    mi = mi.sort_values(ascending=False)
    print(mi)
    mi.plot.bar(figsize=(10, 8))
    plt.ylabel('Mutual Information')
    plt.subplots_adjust(bottom=0.5)


def analyse_ordinal_histogram(feature, path_to_save, dx=3, split_series=False):
    print(f'\n  # HISTOGRAM: {feature}')
    data_dict = multiprocessing.Manager().dict({'A': {},
                                                'B': {},
                                                'C': {},
                                                'D': {}})

    def complexity_entropy(series, driver):
        if split_series:
            series_list = split_data_to_window(series, 300, 60)
        else:
            series_list = [series]
        ordinal_prob_dict = data_dict[driver]
        for _series in series_list:
            _series = preprocessing_to_hc(_series)
            try:
                ord_patterns, probs = ordpy.ordinal_distribution(_series, dx=dx)
                for ord_pattern, prob in zip(ord_patterns, probs):
                    ordinal_prob_dict[str(ord_pattern)] = ordinal_prob_dict.get(str(ord_pattern), []) + [prob]
            except Exception as e:
                print(f'Error in analyse_ordinal_histogram::complexity_entropy. len(_series)={len(_series)}, dx={dx}.',
                      e)
        data_dict[driver] = ordinal_prob_dict

    data_arr = get_data_list()
    process = []
    print(len(data_arr))
    for data in data_arr:
        p = multiprocessing.Process(target=complexity_entropy,
                                    args=(data[0][feature], data[1]))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    #
    fig_name = f'ordinal__{feature}'
    plt.figure(fig_name)
    # TODO: transformar para gráfico de barras, 4 barras para cada ordinal
    for driver, marker in zip(data_dict, ('o', '^', 'd', 'v')):
        ordinal_driver = data_dict[driver]
        x = ordinal_driver.keys()
        y = [np.mean(i) for i in ordinal_driver.values()]
        yerr = [sem(i) for i in ordinal_driver.values()]
        plt.scatter(range(len(x)), y, label=driver, marker=marker)
        # plt.errorbar(x, y, yerr=yerr, fmt='', label=driver, marker=marker)
    plt.legend()
    plt.xlabel('Ordinal')
    plt.ylabel('Probabilities')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f'{path_to_save}/{fig_name}.png')
    return data_dict


def analyse_information_preprocessing():
    data = get_data_list('./ThisCarIsMineInf_window720_dx6')
    df = pd.concat([i[0] for i in data])
    df = df[config.feature_lit_remaining]
    print(df.isna().sum(), '\n')
    print(df.shape)
    print(df.dropna(axis=1).shape)
    s = df.isna().any()
    print(s)


def choose_embedded_dimension(series_length):
    if series_length <= 120:
        return 5
    if series_length <= 720:
        return 6
    if series_length <= 5040:
        return 7
    return 8


def analyse_hc_parameters(path_output):
    def classifier_handle(accuracy, roc_auc):
        try:
            y_pred = cross_val_predict(clf, X, y, cv=5)
            fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
            roc_auc += [metrics.auc(fpr, tpr)]
            accuracy += [metrics.accuracy_score(y, y_pred)]
            # accuracy += list(cross_val_score(clf, X, y, scoring='accuracy'))
            # roc_auc += list(cross_val_score(clf, X, y, scoring='roc_auc'))
        except Exception as e:
            print(f'Error in "analyse_hc_parameters::classifier_handle".', e)

    path_input = []
    X_names = []
    for series_length in range(60, 901, 60):
        d = choose_embedded_dimension(series_length)
        _path = f'./ThisCarIsMineInf_window{series_length}_dx{d}'
        X_names.append(str(series_length))
        path_input.append(_path)

    accuracy_list = []
    accuracy_sem_list = []
    roc_auc_list = []
    roc_auc_sem_list = []
    # classifier = SVC
    classifier = RandomForestClassifier
    print(classifier)
    for path in path_input:
        df_list = []
        for driver in 'ABCD':
            _df = pd.read_csv(path + f'/{driver}/All_1.csv')[config.feature_inf_remaining].dropna()
            _df['driver'] = [driver] * _df.shape[0]
            df_list.append(_df)
        df = pd.concat(df_list)
        print(path, df.shape)
        X = df.drop(['driver'], axis=1)
        manager = multiprocessing.Manager()
        accuracy = manager.list()
        roc_auc = manager.list()
        running_process = []
        for driver in 'ABCD':
            y = df['driver'].replace(['A', 'B', 'C', 'D'],
                                     ['A' == driver, 'B' == driver, 'C' == driver, 'D' == driver])
            clf = classifier()
            p = multiprocessing.Process(target=classifier_handle,
                                        args=(accuracy, roc_auc))
            p.start()
            running_process.append(p)
        for p in running_process:
            p.join()
        accuracy_list.append(mean(accuracy))
        accuracy_sem_list.append(sem(accuracy))
        roc_auc_list.append(mean(roc_auc))
        roc_auc_sem_list.append(sem(roc_auc))
    with open(path_output + '/analyse_data__analyse_hc_parameters.log', 'w') as out:
        json.dump((accuracy_list, accuracy_sem_list, roc_auc_list, roc_auc_sem_list), out)
    with open(path_output + '/analyse_data__analyse_hc_parameters.log', 'r') as data_file:
        accuracy_list, accuracy_sem_list, roc_auc_list, roc_auc_sem_list = json.load(data_file)
        font_size = 24
        plt.figure(figsize=config.default_figsize)
        X_axis = np.arange(len(X_names))
        plt.bar(X_axis - 0.2, accuracy_list, 0.4,
                yerr=accuracy_sem_list, label='Accuracy', color=color1, edgecolor="black")
        plt.bar(X_axis + 0.2, roc_auc_list, 0.4,
                yerr=roc_auc_sem_list, label='ROC AUC', color=color3, edgecolor="black")
        plt.xticks(X_axis, X_names)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.tick_params(axis='x', labelrotation=90)
        plt.subplots_adjust(bottom=.2)
        plt.ylim((0.5, 1))
        # plt.subplots_adjust(bottom=0.2)
        plt.xlabel('Lenght', fontsize=font_size)
        plt.ylabel('Score', fontsize=font_size)
        plt.legend(fontsize=font_size - 4)

        major_ticks = np.arange(.5, 1.01, .1)
        minor_ticks = np.arange(.5, 1.01, .05)
        plt.yticks(major_ticks)
        plt.yticks(minor_ticks, minor=True)
        plt.grid(axis='y')
        plt.grid(which='minor', alpha=0.5)
        plt.grid(which='major', alpha=1)

        plt.savefig(path_output + '/analyse_data__analyse_hc_parameters.png')


def plot_c_limits(dx, path_to_save='.'):
    hc_min, hc_max = hc_limits(dx)
    #
    font_size = 24
    label_size = 26
    fig_name = f'hc_plan__{dx}'
    plt.figure(fig_name, figsize=(10, 8))
    plot_limits(hc_min, hc_max)
    plt.xlabel('Permutation entropy, $H$', fontsize=font_size)
    plt.ylabel('Statistical complexity, $C$', fontsize=font_size)
    plt.tick_params(axis='both', which='major', labelsize=label_size)
    plt.subplots_adjust(bottom=.2)
    plt.savefig(f'{path_to_save}/{fig_name}.png')


def analyse_nan():
    for series_length in range(60, 901, 60):
        d = choose_embedded_dimension(series_length)
        _path = f'./ThisCarIsMineInf_window{series_length}_dx{d}'
        df = get_first_trip(_path)[config.feature_inf_remaining]
        print('\n\n', _path)
        # Calcular a quantidade de NaN em cada coluna
        nan_counts = df.isna().sum()

        # Calcular a porcentagem de NaN em cada coluna
        total_rows = df.shape[0]
        percent_nan = (nan_counts / total_rows) * 100

        # Criar um DataFrame com as informações de NaN e porcentagem de NaN
        nan_info = pd.DataFrame({'NaN Count': nan_counts, 'NaN Percentage': percent_nan})

        # Filtrar as colunas que têm NaN
        nan_info = nan_info[nan_info['NaN Count'] > 0]

        print('\t', 'Número de colunas com NaN:', len(nan_info))
        print('\t', 'Porcentagens:', 'max =', nan_info['NaN Percentage'].max(), 'min =', nan_info['NaN Percentage'].min())

        # print('\t', "Colunas com NaN e porcentagem de NaN:")
        # print('\t', nan_info)
        # df_arr = get_data_list(_path)
        # for df, driver in df_arr:
            # for i, f in enumerate(config.feature_inf_remaining):
            #     print(driver, i + 1,
            #           ', df.shape =', df[f].shape,
            #           ', df_dropna.shape =', df[f].dropna().shape,
            #           ', feature =', f)


def analyse_time(path_output):
    def get_data(path):
        data_arr = [
            pd.read_csv(path + '/A/All_1.csv.time'),
            pd.read_csv(path + '/A/All_2.csv.time'),
            pd.read_csv(path + '/A/All_3.csv.time'),
            pd.read_csv(path + '/A/All_4.csv.time'),
            pd.read_csv(path + '/A/All_5.csv.time'),
            #
            pd.read_csv(path + '/B/All_1.csv.time'),
            pd.read_csv(path + '/B/All_2.csv.time'),
            pd.read_csv(path + '/B/All_3.csv.time'),
            pd.read_csv(path + '/B/All_4.csv.time'),
            pd.read_csv(path + '/B/All_6.csv.time'),
            #
            pd.read_csv(path + '/C/All_1.csv.time'),
            pd.read_csv(path + '/C/All_2.csv.time'),
            pd.read_csv(path + '/C/All_3.csv.time'),
            pd.read_csv(path + '/C/All_4.csv.time'),
            pd.read_csv(path + '/C/All_5.csv.time'),
            #
            pd.read_csv(path + '/D/All_1.csv.time'),
            pd.read_csv(path + '/D/All_3.csv.time'),
            pd.read_csv(path + '/D/All_5.csv.time'),
            pd.read_csv(path + '/D/All_6.csv.time'),
            pd.read_csv(path + '/D/All_7.csv.time'),
        ]
        return pd.concat(data_arr)

    time_lit_20 = get_data('./ThisCarIsMineNormalized_20')['time_lit'].to_list()
    time_lit_120 = get_data('./ThisCarIsMineNormalized_120')['time_lit'].to_list()
    time_lit_300 = get_data('./ThisCarIsMineNormalized_300')['time_lit'].to_list()
    time_lit_720 = get_data('./ThisCarIsMineNormalized_720')['time_lit'].to_list()
    time_lit_900 = get_data('./ThisCarIsMineNormalized_900')['time_lit'].to_list()
    time_inf_20 = get_data('./ThisCarIsMineInf_window20_dx4')['time_hc'].to_list()
    time_inf_120 = get_data('./ThisCarIsMineInf_window120_dx5')['time_hc'].to_list()
    time_inf_300 = get_data('./ThisCarIsMineInf_window300_dx6')['time_hc'].to_list()
    time_inf_720 = get_data('./ThisCarIsMineInf_window720_dx6')['time_hc'].to_list()
    time_inf_900 = get_data('./ThisCarIsMineInf_window900_dx7')['time_hc'].to_list()
    #
    lit_mean = [
        mean(time_lit_20),
        mean(time_lit_120),
        mean(time_lit_300),
        mean(time_lit_720),
        mean(time_lit_900)
    ]
    lit_sem = [
        sem(time_lit_20),
        sem(time_lit_120),
        sem(time_lit_300),
        sem(time_lit_720),
        sem(time_lit_900)
    ]
    inf_mean = [
        mean(time_inf_20),
        mean(time_inf_120),
        mean(time_inf_300),
        mean(time_inf_720),
        mean(time_inf_900)
    ]
    inf_sem = [
        sem(time_inf_20),
        sem(time_inf_120),
        sem(time_inf_300),
        sem(time_inf_720),
        sem(time_inf_900)
    ]
    with open(path_output + '/analyse_data__compute_time.log', 'w') as out:
        json.dump((lit_mean, lit_sem, inf_mean, inf_sem), out)
    font_size = 24
    plt.figure(figsize=config.default_figsize)
    X_names = [
        '20',
        '120',
        '300',
        '720',
        '900'
    ]
    with open(path_output + '/analyse_data__compute_time.log', 'r') as data_file:
        lit_mean, lit_sem, inf_mean, inf_sem = json.load(data_file)
        X_axis = np.arange(len(X_names))
        plt.bar(X_axis - 0.2, lit_mean, 0.4,
                yerr=lit_sem, label='Literature', color=color1, edgecolor="black")
        plt.bar(X_axis + 0.2, inf_mean, 0.4,
                yerr=inf_sem, label='Proposal', color=color3, edgecolor="black")
        plt.xticks(X_axis, X_names)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        # plt.ylim((0.5, 1))
        # plt.subplots_adjust(bottom=0.1)
        plt.xlabel('Lenght', fontsize=font_size)
        plt.ylabel('Time (s)', fontsize=font_size)
        plt.legend(fontsize=font_size - 4)
        plt.savefig(path_output + '/analyse_data__compute_time.png')
        for lit, inf in zip(lit_mean, inf_mean):
            print(f'lit: {lit}, inf: {inf}. ', inf - lit, inf / lit)


def sliding_window_plt():
    feature = 'engine_torque'
    df = pd.read_csv('./ThisCarIsMine/A/All_1.csv')[:300]
    df = df[feature]
    font_size = 24
    plt.figure(figsize=config.default_figsize)
    plt.plot(df)
    plt.xlabel('Time (s)', fontsize=font_size)
    # plt.ylabel('Time (s)', fontsize=font_size)
    plt.xticks(range(0, 301, 30))
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.tick_params(left=False, right=False, labelleft=False)
    plt.savefig('sliding_window_plt.png')


def analyse_inf_feature_nan(features):
    data_arr = get_data_list('./ThisCarIsMineInf_window720_dx6')
    features_inf = []
    for i in features:
        features_inf.append(f'{i}_entropy')
        features_inf.append(f'{i}_complexity')
        features_inf.append(f'{i}_fisher')
    df = pd.concat([i[0] for i in data_arr])[features_inf]
    columns_with_nan = df.columns[df.isna().any()].tolist()
    remaining = list(set(features_inf) - set(columns_with_nan))

    # Calcular a quantidade de NaN em cada coluna
    nan_counts = df.isna().sum()

    # Calcular a porcentagem de NaN em cada coluna
    total_rows = df.shape[0]
    percent_nan = (nan_counts / total_rows) * 100

    # Criar um DataFrame com as informações de NaN e porcentagem de NaN
    nan_info = pd.DataFrame({'NaN Count': nan_counts, 'NaN Percentage': percent_nan})

    # Filtrar as colunas que têm NaN
    nan_info = nan_info[nan_info['NaN Count'] > 0]
    # print("Colunas com NaN e porcentagem de NaN:")
    # print(nan_info)

    return remaining, columns_with_nan


def analyse_inf_plane(feature, path_to_save, plane, dx=7, split_series=False):
    print(f'\n  # Information Plane: {feature}')
    data_dict = multiprocessing.Manager().dict(
        {
            'A': ([], [], [], []),  # Entropy, Complexity, Fisher, Shannon
            'B': ([], [], [], []),
            'C': ([], [], [], []),
            'D': ([], [], [], [])
        }
    )

    def information_measure(series, driver, feature):
        if split_series:
            series_list = split_data_to_window(series, 120, 60)
        else:
            series_list = [series]
        h_list_local, c_list_local, f_list_local, s_list_local = [], [], [], []
        for _series in series_list:
            _series = preprocessing_to_hc(_series)
            try:
                if 'hc' in plane:
                    h, c = ordpy.complexity_entropy(_series, dx=dx)
                    h_list_local.append(h)
                    c_list_local.append(c)
                if 'fs' in plane:
                    s, f = ordpy.fisher_shannon(_series, dx=dx)
                    f_list_local.append(f)
                    s_list_local.append(s)
            except Exception as e:
                print(f'Error in analyse_inf_plan::information_measure. '
                      f'len(_series)={len(_series)}, dx={dx}, feature={feature}.', e)
        h_list, c_list, f_list, s_list = data_dict[driver]
        data_dict[driver] = (h_list+h_list_local, c_list+c_list_local, f_list+f_list_local, s_list+s_list_local)

    data_arr = get_data_list()
    process = []
    for data in data_arr:
        p = multiprocessing.Process(target=information_measure,
                                    args=(data[0][feature], data[1], feature))
        p.start()
        process.append(p)
    print('Number of processed files:', len(process))
    for p in process:
        p.join()
    with open('data_dict.json', 'w') as file:
        json.dump(dict(data_dict), file)
    hc_min, hc_max = hc_limits(dx)
    #
    font_size = 24
    legend_font_size = 18
    label_size = 26
    # index for dictionary
    H, C, F, S = 0, 1, 2, 3  # Entropy, Complexity, Fisher, Shannon
    #
    if 'hc' in plane:
        fig_name = f'hc_plan__{feature}'
        plt.figure(fig_name, figsize=config.default_figsize)
        for driver, marker in zip('ABCD', 'o^dv'):
            h_list = data_dict[driver][H]
            c_list = data_dict[driver][C]
            limit = min(len(h_list), len(c_list))
            plt.scatter(h_list[:limit], c_list[:limit], label=f'driver {driver}', s=100, marker=marker)
        plot_limits(hc_min, hc_max)
        plt.legend(fontsize=legend_font_size)
        plt.xlabel('Permutation entropy, $H$', fontsize=font_size)
        plt.ylabel('Statistical complexity, $C$', fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=label_size)
        plt.subplots_adjust(bottom=.2, left=.2)
        plt.savefig(f'{path_to_save}/{fig_name}.png')
        #
        #
        fig_name = f'hc_plan__{feature}__zoom'
        plt.figure(fig_name, figsize=config.default_figsize)
        for driver, marker in zip('ABCD', 'o^dv'):
            h_list = data_dict[driver][H]
            c_list = data_dict[driver][C]
            limit = min(len(h_list), len(c_list))
            plt.scatter(h_list[:limit], c_list[:limit], label=f'driver {driver}', s=100, marker=marker)
        plt.legend(fontsize=legend_font_size)
        plt.xlabel('Permutation entropy, $H$', fontsize=font_size)
        plt.ylabel('Statistical complexity, $C$', fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=label_size)
        plt.subplots_adjust(bottom=.2, left=.2)
        plt.savefig(f'{path_to_save}/{fig_name}.png')
    #
    if 'fs' in plane:
        fig_name = f'fs_plan__{feature}'
        plt.figure(fig_name, figsize=config.default_figsize)
        for driver, marker in zip('ABCD', 'o^dv'):
            f_list = data_dict[driver][F]
            s_list = data_dict[driver][S]
            limit = min(len(f_list), len(s_list))
            plt.scatter(s_list[:limit], f_list[:limit], label=f'driver {driver}', s=100, marker=marker)
        plt.legend(fontsize=legend_font_size)
        plt.ylim((0, 1))
        plt.xlim((0, 1))
        plt.xlabel('Shannon entropy, $S$', fontsize=font_size)
        plt.ylabel('Fisher entropy, $F$', fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=label_size)
        plt.subplots_adjust(bottom=.2, left=.2)
        plt.savefig(f'{path_to_save}/{fig_name}.png')
        #
        #
        fig_name = f'fs_plan__{feature}__zoom'
        plt.figure(fig_name, figsize=config.default_figsize)
        for driver, marker in zip('ABCD', 'o^dv'):
            f_list = data_dict[driver][F]
            s_list = data_dict[driver][S]
            limit = min(len(f_list), len(s_list))
            plt.scatter(s_list[:limit], f_list[:limit], label=f'driver {driver}', s=100, marker=marker)
        plt.legend(fontsize=legend_font_size)
        plt.xlabel('Shannon entropy, $S$', fontsize=font_size)
        plt.ylabel('Fisher entropy, $F$', fontsize=font_size)
        plt.tick_params(axis='both', which='major', labelsize=label_size)
        plt.subplots_adjust(bottom=.2, left=.2)
        plt.savefig(f'{path_to_save}/{fig_name}.png')
    return data_dict


if __name__ == '__main__':
    # # INDIFFERENT
    # print('# INDIFFERENT')
    # features_indifferent = analyse_indifferent()
    # features_remaining = list(set(config.ALL_FEATURES) - set(features_indifferent))
    # print('[!] Features indifferent:', len(features_indifferent), features_indifferent)
    # print('[!] Features remaining:', len(features_remaining), features_remaining)
    #
    # # VARIANCE
    # print('# VARIANCE')
    # feature_invariance = analyse_variance(features_remaining)
    # features_remaining = list(set(features_remaining) - set(feature_invariance))
    # print('[!] Features invariance:', len(feature_invariance), feature_invariance)
    # print('[!] Features remaining:', len(features_remaining), features_remaining)
    #
    # # data_arr = get_data_list()
    # # df = pd.concat([i[0] for i in data_arr])[features_remaining]
    # df = get_first_trip()[features_remaining]
    #
    # # CORRELATION
    # print('# CORRELATION')
    # corr, inc, exc = analyse_correlation(df=df, correlate_threshold=.90)
    # print('[!] Features correlation:', len(exc), exc)
    # print('[!] Features remaining:', len(inc), inc)
    # plot_correlation(corr)
    #
    # # Removing feature NaN
    # print('# Removing feature NaN')
    # inc, exc = analyse_inf_feature_nan(inc)
    # print('[!] Features inf NaN:', len(exc), exc)
    # print('[!] Features remaining:', len(inc), inc)

    # # INF THEORY PLAN
    # dx = 7
    # path_to_save = 'results/inf_plane'
    # for feature in [
    #     'accelerator_position',
    #     'steering_wheel_angle',
    #     'car_speed',
    #     'cooling_temperature',
    #     # 'inhale_pressure',
    #     # 'engine_torque',
    #     # 'inhale_pressure',
    #     # 'throttle_position_abs',
    #     # 'long_fuel_bank',
    #     # 'engine_speed',
    #     # 'friction_torque',
    #     # 'engine_torque',
    # ]:
    #     analyse_inf_plane(feature, path_to_save, plane='hcfs', dx=dx)
    # plt.show()

    # analyse_hc_parameters('results/hc_plan')

    # analyse_time('results')
    # sliding_window_plt()
    # plot_c_limits(3, path_to_save)
    # plot_c_limits(5, path_to_save)
    # plot_c_limits(6, path_to_save)

    # HISTOGRAM
    # for feature in ['steering_wheel_angle',
    #                 'inhale_pressure',
    #                 'engine_torque',
    #                 'accelerator_position',
    #                 'inhale_pressure',
    #                 'throttle_position_abs',
    #                 'long_fuel_bank',
    #                 'engine_speed',
    #                 'friction_torque',
    #                 'cooling_temperature',
    #                 'engine_torque',
    #                 'car_speed']:
    #     analyse_ordinal_histogram(feature, path_to_save, dx=dx)

    # plt.show()

    analyse_nan()
