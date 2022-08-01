#!/usr/bin/python3

SAMPLE_SIZE=30

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from random import randint

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

TARGET = 'label'

# GET DATA

def get_sample(samplesize=30, TARGET='label'):
    data_a = pd.read_csv('../A/All_1.csv')
    data_b = pd.read_csv('../B/All_1.csv')
    data_c = pd.read_csv('../C/All_1.csv')
    data_d = pd.read_csv('../D/All_1.csv')
    data_a = data_a.sample(samplesize*3)
    data_a[TARGET] = [1] * data_a.shape[0]
    data_b = data_b.sample(samplesize)
    data_b[TARGET] = [0] * data_b.shape[0]
    data_c = data_c.sample(samplesize)
    data_c[TARGET] = [0] * data_c.shape[0]
    data_d = data_d.sample(samplesize)
    data_d[TARGET] = [0] * data_d.shape[0]
    return pd.concat([data_a, data_b, data_c, data_d], axis=0)
    #return data_a, data_b, data_c, data_d


# PRE-PROCESSING

def remove_correlation(df, TARGET='label'):
    correlate_threshold = 0.95
    included = [TARGET]
    columns = list(df.drop([TARGET], axis=1).columns)
    for i in range(len(columns)):
        c1 = df[columns[i]]
        must_add = True
        for j in range(i+1, len(columns), 1):
            c2 = df[columns[j]]
            if c1.corr(c2) > 0.95:
                must_add = False
                break
        if must_add:
            included.append(columns[i])   
    return df[included]

def remove_miss_value(df):
    return df.dropna(axis=0)

def normalization(df):
    x = df.drop([TARGET], axis=1).values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    new_df = pd.DataFrame(x_scaled, columns=df.drop([TARGET], axis=1).columns)
    new_df[TARGET] = list(df[TARGET])
    return new_df

def remove_invariants(df):
    return df.loc[:, (df != df.iloc[0]).any()]

def window(df, size):
    new_df = df.rolling(size, center=True).mean()
    new_df[TARGET] = df[TARGET]
    return new_df

def dataprocessing(df, WINDOW=30):
    df = remove_correlation(df)
    #df = df[features+[TARGET]]
    print(" - Remove highly-correlated, features:", '\t', df.shape)
    df = remove_invariants(df)
    print(" - Remove invariants, features:", '\t', df.shape)
    df = normalization(df)
    print(" - Normaliation", '\t', df.shape)
    df = window(df, WINDOW)
    print(f" - Apply mean window of {WINDOW} seconds", '\t', df.shape)
    df = remove_miss_value(df)
    print(" - Remove NaN:", '\t', df.shape)
    return df

if __name__ == '__main__':
    def df_to_tuple(df, features):
        arr1 = tuple(zip(df[features[0]], df[features[1]]))
        arr2 = tuple(df[TARGET])
        return (arr1, arr2)

    df = dataprocessing(get_sample(SAMPLE_SIZE))
    columns = df.drop([TARGET], axis=1).columns

    pair_fetures = [
        [columns[randint(0, len(columns)-1)], columns[randint(0, len(columns)-1)]],
        [columns[randint(0, len(columns)-1)], columns[randint(0, len(columns)-1)]],
        [columns[randint(0, len(columns)-1)], columns[randint(0, len(columns)-1)]],
        [columns[randint(0, len(columns)-1)], columns[randint(0, len(columns)-1)]],
    ]

    datasets = [df_to_tuple(df, p) for p in pair_fetures]

    # PCA dataset
    encoder = LabelEncoder()
    X = df.drop([TARGET], axis=1)
    y_label = df[TARGET]
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X)
    pca2 = PCA(n_components=2)
    pca2.fit(X_features)
    x_pca2 = pca2.transform(X_features)
    pair_fetures.append(['pca1', 'pca2'])

    datasets.append([x_pca2, y_label])

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel(pair_fetures[ds_cnt][0])
        ax.set_ylabel(pair_fetures[ds_cnt][1])
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.6,
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1

    plt.tight_layout()
    plt.show()