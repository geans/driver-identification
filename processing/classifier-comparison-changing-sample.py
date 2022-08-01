#!/usr/bin/python3

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
    if samplesize > 0:
        samplesize = samplesize//6
        data_a = data_a.sample(samplesize*3)
        data_a[TARGET] = [1] * data_a.shape[0]
        data_b = data_b.sample(samplesize)
        data_b[TARGET] = [0] * data_b.shape[0]
        data_c = data_c.sample(samplesize)
        data_c[TARGET] = [0] * data_c.shape[0]
        data_d = data_d.sample(samplesize)
        data_d[TARGET] = [0] * data_d.shape[0]
    else:
        samplesize = min(data_a.shape[0], data_b.shape[0], data_c.shape[0], data_d.shape[0])
        data_a = data_a[:samplesize]
        data_a[TARGET] = [1] * data_a.shape[0]
        data_b = data_b[:samplesize//3]
        data_b[TARGET] = [0] * data_b.shape[0]
        data_c = data_c[:samplesize//3]
        data_c[TARGET] = [0] * data_c.shape[0]
        data_d = data_d[:samplesize//3]
        data_d[TARGET] = [0] * data_d.shape[0]
    return pd.concat([data_a, data_b, data_c, data_d], axis=0)


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
    new_df = df.rolling(size, center=True, min_periods=1).mean()
    new_df[TARGET] = df[TARGET]
    return new_df

def dataprocessing(df, WINDOW=30):
    df = normalization(df)
    #print(" - Normaliation", '\t', df.shape)
    df = remove_correlation(df)
    #print(" - Remove highly-correlated, features:", '\t', df.shape)
    df = remove_invariants(df)
    #print(" - Remove invariants, features:", '\t', df.shape)
    df = window(df, WINDOW)
    #print(f" - Apply mean window of {WINDOW} seconds", '\t', df.shape)
    df = remove_miss_value(df)
    #print(" - Remove NaN:", '\t', df.shape)
    return df


if __name__ == '__main__':
    INIT_SIZE, MAX_SIZE, STEP = 30, 1020, 60

    fig, ax = plt.subplots(1, 2, figsize=(27,9))

    i = 0
    # iterate over classifiers
    print("\nRaw Data\n")
    for name, clf in zip(names, classifiers):
        x = []
        scores = []
        print("Classifier:", name)
        for samplesize in range(INIT_SIZE, MAX_SIZE, STEP):
            df = get_sample(samplesize)

            x.append(samplesize)
            X, y = df.drop([TARGET], axis=1), df[TARGET]
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=42
            )

            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        ax[i-1].plot(x, scores, label=name)
    ax[i-1].set_title('raw')

    i += 1
    # iterate over classifiers
    print("\nProcessed Data\n")
    for name, clf in zip(names, classifiers):
        x = []
        scores = []
        print("Classifier:", name)
        for samplesize in range(INIT_SIZE, MAX_SIZE, STEP):
            df_raw = get_sample(samplesize)
            df = dataprocessing(df_raw)

            x.append(samplesize)
            X, y = df.drop([TARGET], axis=1), df[TARGET]
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        ax[i-1].plot(x, scores, label=name)
    ax[i-1].set_title('processed')


    plt.tight_layout()
    plt.legend()
    plt.show()