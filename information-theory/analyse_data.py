import multiprocessing
import time
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import sem
import seaborn as sns

import config
import data_preprocessing as dpp
from information import InformationHandle
from getdata import GetData


def getx(df):
    return df.drop([config.label, config.driver], axis=1)


def analyse_correlation(df, correlate_threshold):
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

    print('Correlation')
    print(f'  Negative range: [{min(mini)}, {max(mini)}]')
    print(f'  Positive range: [{min(maxi)}, {max(maxi)}]')
    print()

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
    print('correlation =', correlate_threshold, '\n')
    print('included =', included, '#', len(included), '\n')
    print('excluded =', excluded, '#', len(excluded), '\n')

    print(corr)
    return corr, included, excluded


def plot_correlation(corr):
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
    cb.ax.tick_params(labelsize=14)
    plt.tick_params(labelsize=14)


def analyse_variance(dfx):
    variance_df = dfx.loc[:, (dfx != dfx.iloc[0]).any()]
    invariance = list(set(dfx.columns) - set(variance_df.columns))
    print('feature_invariance =', invariance, '#', len(invariance))
    print('feature_variance =', list(variance_df.columns), '#', variance_df.shape[1])
    return invariance, variance_df


if __name__ == '__main__':
    program_time = time.time()

    features = config.ALL_FEATURES
    # features = config.features

    data = GetData(path_dataset=config.path_dataset,
                   label_feature_name=config.label,
                   driver_feature_name=config.driver,
                   features=features,
                   trips=None)  # get all trips
    df = data.get_all(driver_target='A')

    # Analyse
    _, variance_df = analyse_variance(getx(df))
    analyse_correlation(df=variance_df, correlate_threshold=.95)

    delta = str(timedelta(seconds=(time.time() - program_time)))
    print('\n[time]', delta)

    plt.show()
