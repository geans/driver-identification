#!/usr/bin/python3

import config
from plotresults import PlotResults

PATH_DATASET = config.path_dataset
SAMPLE_SIZE = config.sample_size
HC_SIZE = config.hc_size
NUM_REPETITIONS = config.num_repetitions
dx,dy,tx,ty = config.entropy_complexity_parameters
features = config.features
DEBUG_ON_SCREEN = config.debug_on_screen

import pandas as pd
import time
import ordpy
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from random import randint
from scipy.stats import sem
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from threading import Thread
import multiprocessing

import warnings
warnings.filterwarnings("ignore")

# Create empty file
f = open('output.txt', 'w')
f.close()

def mdebug(*objects, sep=' ', end='\n', file=None, flush=False):
    if DEBUG_ON_SCREEN:
        print(*objects, sep=sep, end=end, file=file, flush=flush)
    f = open('output.txt', 'a')
    for obj in objects:
        f.write(str(obj))
        f.write(sep)
    f.write(end)
    f.close()

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

TARGET = 'label'
DRIVER = 'driver'


# GET DATA

def get_sample(path_dataset, samplesize=300, TARGET='label', driver='A'):
    list_df_a = []
    list_df_b = []
    list_df_c = []
    list_df_d = []
    # read all files
    for i in range(1,1+8): list_df_a.append(pd.read_csv(path_dataset + f'A/All_{i}.csv'))
    for i in range(1,1+8): list_df_b.append(pd.read_csv(path_dataset + f'B/All_{i}.csv'))
    for i in range(1,1+5): list_df_c.append(pd.read_csv(path_dataset + f'C/All_{i}.csv'))
    for i in range(1,1+9): list_df_d.append(pd.read_csv(path_dataset + f'D/All_{i}.csv'))
    data_a = pd.concat(list_df_a)
    data_b = pd.concat(list_df_b)
    data_c = pd.concat(list_df_c)
    data_d = pd.concat(list_df_d)
    #
    if samplesize > 0:
        i = randint(0, max(data_a.shape[0] - samplesize, 0))
        data_a = data_a[i:i+samplesize]
        i = randint(0, max(data_b.shape[0] - samplesize, 0))
        data_b = data_b[i:i+samplesize]
        i = randint(0, max(data_c.shape[0] - samplesize, 0))
        data_c = data_c[i:i+samplesize]
        i = randint(0, max(data_d.shape[0] - samplesize, 0))
        data_d = data_d[i:i+samplesize]
    #
    data_a[TARGET] = [int(driver=='A')] * data_a.shape[0]
    data_b[TARGET] = [int(driver=='B')] * data_b.shape[0]
    data_c[TARGET] = [int(driver=='C')] * data_c.shape[0]
    data_d[TARGET] = [int(driver=='D')] * data_d.shape[0]
    data_a[DRIVER] = ['A'] * data_a.shape[0]
    data_b[DRIVER] = ['B'] * data_b.shape[0]
    data_c[DRIVER] = ['C'] * data_c.shape[0]
    data_d[DRIVER] = ['D'] * data_d.shape[0]
    #
    return [data_a[features+[DRIVER,TARGET]],
           data_b[features+[DRIVER,TARGET]],
           data_c[features+[DRIVER,TARGET]],
           data_d[features+[DRIVER,TARGET]]]


# PRE-PROCESSING

def remove_correlation(df, TARGET='label'):
    correlate_threshold = 0.95
    included = [TARGET, DRIVER]
    columns = list(df.drop([TARGET, DRIVER], axis=1).columns)
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
    x = df.drop([TARGET, DRIVER], axis=1).values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    new_df = pd.DataFrame(x_scaled, columns=df.drop([TARGET, DRIVER], axis=1).columns)
    new_df[TARGET] = list(df[TARGET])
    new_df[DRIVER] = list(df[DRIVER])
    return new_df

def remove_invariants(df):
    new_df = df.drop([TARGET, DRIVER], axis=1)
    new_df.loc[:, (new_df != new_df.iloc[0]).any()]
    new_df[TARGET] = list(df[TARGET])
    new_df[DRIVER] = list(df[DRIVER])
    return new_df

def window(df, size):
    new_df = df.rolling(size, center=True, min_periods=1).mean()
    new_df[TARGET] = df[TARGET]
    new_df[DRIVER] = df[DRIVER]
    return new_df

def dataprocessing(df_list, WINDOW=33):
    df = pd.concat(df_list)
    mdebug('[!] describe:' , df[[DRIVER, TARGET]])
    df = normalization(df)
    mdebug('[!] normalized:', df[[DRIVER, TARGET]].shape)
    mdebug('[!] describe:' , df[[DRIVER, TARGET]])
    df = remove_correlation(df)
    mdebug('[!] correçation:', df[[DRIVER, TARGET]].shape)
    mdebug('[!] describe:' , df[[DRIVER, TARGET]])
    df = remove_invariants(df)
    mdebug('[!] invariant:', df[[DRIVER, TARGET]].shape)
    # df = window(df, WINDOW)
    # mdebug('[!] window:', df[[DRIVER, TARGET]].shape)
    mdebug('[!] describe:' , df[[DRIVER, TARGET]])
    df = remove_miss_value(df)
    mdebug('[!] miss:', df[[DRIVER, TARGET]].shape)
    return df


# HANDLE TIME SERIES

def get_sublists(original_list, delta):
    pivot = 0
    sublists = []
    len_list = len(original_list)
    shift = 1
    while pivot+delta <= len_list:
        sublists.append(original_list[pivot:pivot+delta])
        pivot += shift
    return sublists

def get_information(df_list, driver='A', dx=3, dy=1, taux=1, tauy=1, both=False):
    y_label = []
    X = []
    new_df_list = []
    for df, label in zip(df_list, ['A', 'B', 'C', 'D']):
        sliding_window_df = get_sublists(df, HC_SIZE)
        new_df = None
        for window_df in sliding_window_df:
            row = {}
            for feature in window_df.drop([TARGET, DRIVER], axis=1).columns:
                h, c = ordpy.complexity_entropy(window_df[feature], dx=dx, dy=dy, taux=taux, tauy=tauy)
                f, s = ordpy.fisher_shannon(window_df[feature], dx=4)
                row[f'{feature}_entropy'] = h
                row[f'{feature}_complexity'] = c
                row[f'{feature}_fisher'] = f
                row[f'{feature}_shannon'] = s
                if both:
                    if window_df[f].dtypes == int:
                        row[f] = int(window_df[f].head(1))
                    else:
                        row[f] = float(window_df[f].head(1))
            row[TARGET] = int(driver==label)
            row[DRIVER] = driver
            # print('Row:', row)
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                new_df.loc[len(new_df)] = row
        # new_df[TARGET] = df[TARGET]
        # new_df[DRIVER] = df[DRIVER]
        new_df_list.append(new_df)

    mdebug('[!] get information entry:' , pd.concat(df_list)[[DRIVER, TARGET]])
    mdebug('[!] get information out:  ' , pd.concat(new_df_list)[[DRIVER, TARGET]])
    return new_df_list


# SEARCH BESTS PARAMETERS

def best_parameter(df):
    mdebug('Calculing best parameter to HC.')
    param = 'parameter'
    best_score = 0
    best_param = []

    _1to6 = [1,2,3,4,5,6]

    for dx in _1to6:
        for dy in [1]:
            for tx in _1to6:
                for ty in [1,2,3]:
                    # clf = SVC()
                    # clf = KNeighborsClassifier(10)
                    clf = SVC(gamma=2, C=1)
                    score_inf_teory = []
                    try:
                        for driver_target in ['A', 'B', 'C', 'D']:
                            X, y = get_information(
                                df,
                                driver=driver_target,
                                dx=dx,
                                dy=dy,
                                taux=tx,
                                tauy=ty
                            )
                            X_train, X_test, y_train, y_test = train_test_split(X, y)
                            clf.fit(X_train, y_train)
                            score = clf.score(X_test, y_test)
                            score_inf_teory.append(score)
                        score_inf_teory = sum(score_inf_teory) / len(score_inf_teory)
                        # mdebug(' ', score_inf_teory)
                        if score_inf_teory > best_score:
                            best_score = score_inf_teory
                            best_param = [dx, dy, tx, ty]
                    except Exception as e:
                        mdebug('[!] Error to paramaters:', [dx, dy, tx, ty], '\b.')#, e)
    mdebug('Best parameters:', best_param, best_score)
    return best_param

def classifiers_train_test(X, y, _score_dic, _time_dic, num_repetitions):
    running_process = []
    manager = multiprocessing.Manager()
    score_dic = manager.dict(_score_dic)
    time_dic= manager.dict(_time_dic)
    def classifier_handle(name, clf, X_train, X_test, y_train, y_test):
        # print('Init classifier:', name)
        t0 = time.time()
        clf.fit(X_train, y_train)
        time_fit = time.time() - t0
        score = clf.score(X_test, y_test)
        score_dic[name] = score_dic.get(name, []) + [score]
        time_dic[name] = time_dic.get(name, []) + [time_fit]
        # print('End classifier:', name)
    for _ in range(num_repetitions):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        for name, clf in zip(classifier_names, classifiers):
            p = multiprocessing.Process(target=classifier_handle,
                                        args=(name, clf, X_train, X_test, y_train, y_test))
            p.start()
            # p.join()
            running_process.append(p)
        for p in running_process:
            p.join()
    mdebug(f'''
    [partial data]
        score_dic = {score_dic}
        time_dic  = {time_dic}
    [end partial data]
    ''')
    return score_dic, time_dic


if __name__ == '__main__':
    time_literature = {}
    score_literature = {}
    time_dataprocessing = []

    time_inf_teory = {}
    score_inf_teory = {}
    time_hc_calculate = []

    time_both = {}
    score_both = {}
    time_data_both = []

    mdebug(
f'''
Sample size: {SAMPLE_SIZE}
HC size: {HC_SIZE}
Number of repetitions: {NUM_REPETITIONS}
dx,dy,tx,ty = {dx,dy,tx,ty}
'''
    )

    for driver_target in ['A', 'B', 'C', 'D']:
        df_raw = get_sample(PATH_DATASET, SAMPLE_SIZE, driver=driver_target)
        mdebug('Dataset size shape:', pd.concat(df_raw).shape)
        mdebug('\nDriver', driver_target, '\n')


        # Data process

        t0 = time.time()
        df_pp = dataprocessing(df_raw)
        df_pp = window(df_pp, 33)
        time_dataprocessing.append(time.time() - t0)
        mdebug('[!] window:', df_pp.shape)
        mdebug('[!][!] describe:' , df_pp[[DRIVER, TARGET]].describe())
        mdebug('[!][!] describe:' , df_pp[[DRIVER, TARGET]])

        X, y = df_pp.drop([TARGET, DRIVER], axis=1), df_pp[TARGET]
        mdebug(f"Literature.", df_pp.drop([TARGET, DRIVER], axis=1).shape[1],
                f'Features after pre-processing:',
                list(df_pp.drop([TARGET, DRIVER], axis=1).columns))
        score_literature, time_literature = classifiers_train_test(X, y, score_literature, 
                                                                    time_literature, 
                                                                    NUM_REPETITIONS)

        # Entropy-Complexty (HC)

        t0 = time.time()
        df_hc = get_information(
            df_raw,
            driver=driver_target,
            dx=dx,
            dy=dy,
            taux=tx,
            tauy=ty
        )
        time_hc_calculate.append(time.time() - t0)
        mdebug(f"Information-Teory.")

        df_joined = pd.concat(df_hc)
        X, y = df_joined.drop([TARGET, DRIVER], axis=1), df_joined[TARGET]
        score_inf_teory, time_inf_teory = classifiers_train_test(X, y, score_inf_teory, 
                                                                    time_inf_teory, 
                                                                    NUM_REPETITIONS)


        # Entropy-Complexty (HC) + Literature

        mdebug(f"Both.")
        df_both = dataprocessing(df_hc)

        df_joined = df_both
        X, y = df_joined.drop([TARGET, DRIVER], axis=1), df_joined[TARGET]
        mdebug(f"Literature.", df_joined.drop([TARGET, DRIVER], axis=1).shape[1],
                f'Features after pre-processing:',
                list(df_joined.drop([TARGET, DRIVER], axis=1).columns))
        score_both, time_both = classifiers_train_test(X, y, score_both, 
                                                        time_both,
                                                        NUM_REPETITIONS)


    # chart of score

    y_lt_accuracy             = []
    y_lt_std_error_accuracy   = []
    y_inf_accuracy            = []
    y_inf_std_error_accuracy  = []
    y_both_accuracy           = []
    y_both_std_error_accuracy = []

    y_lt_trainingtime             = []
    y_lt_std_error_trainingtime   = []
    y_inf_trainingtime            = []
    y_inf_std_error_trainingtime  = []
    y_both_trainingtime           = []
    y_both_std_error_trainingtime = []

    # time_dataprocessing = []
    # time_hc_calculate   = []

    y_lt_totaltime             = []
    y_lt_std_error_totaltime   = []
    y_inf_totaltime            = []
    y_inf_std_error_totaltime  = []
    y_both_totaltime           = []
    y_both_std_error_totaltime = []


    mdebug('\nChart')
    
    for key in classifier_names:
        y_lt_accuracy.append(mean(score_literature[key]))
        y_lt_std_error_accuracy.append(sem(score_literature[key]))
        
        y_inf_accuracy.append(mean(score_inf_teory[key]))
        y_inf_std_error_accuracy.append(sem(score_inf_teory[key]))
        
        y_both_accuracy.append(mean(score_both[key]))
        y_both_std_error_accuracy.append(sem(score_both[key]))

    X_axis = np.arange(len(classifier_names))
    mdebug(f'''
[data]  Classifier's Accuracy:

- Literature:
  - accuracy:  {y_lt_accuracy}
  - std error: {y_lt_std_error_accuracy}

- Complexity-Entropy:
  - accuracy:  {y_inf_accuracy}
  - std error: {y_inf_std_error_accuracy}

- Both:
  - accuracy:  {y_both_accuracy}
  - std error: {y_both_std_error_accuracy}

[end data]
        ''')
    
    mdebug('Output training times (ms):')
    for key in classifier_names:
        y_lt_trainingtime.append(mean(time_literature[key]) * 1000)
        y_lt_std_error_trainingtime.append(sem(time_literature[key]) * 1000)
        
        y_inf_trainingtime.append(mean(time_inf_teory[key]) * 1000)
        y_inf_std_error_trainingtime.append(sem(time_inf_teory[key]) * 1000)
        
        y_both_trainingtime.append(mean(time_both[key]) * 1000)
        y_both_std_error_trainingtime.append(sem(time_both[key]) * 1000)

        mdebug(f'''
    {key}:
      - Literature  : {y_lt_trainingtime[-1]} +- {y_lt_std_error_trainingtime[-1]}
      - Comp.Entropy: {y_inf_trainingtime[-1]} +- {y_inf_std_error_trainingtime[-1]}
      - Both        : {y_both_trainingtime[-1]} +- {y_both_std_error_trainingtime[-1]}
''')

    mdebug(f'''
[data]  Classifier's Training time (ms):

- Literature:
  y_lt_trainingtime = {y_lt_trainingtime}
  y_lt_std_error_trainingtime = {y_lt_std_error_trainingtime}
  y_inf_trainingtime = {y_inf_trainingtime}
  y_inf_std_error_trainingtime = {y_inf_std_error_trainingtime}
  y_both = {y_both_trainingtime}
  y_both_std_error_trainingtime = {y_both_std_error_trainingtime}

[end data]
        ''')

    mdebug(f'''
[data]  Classifier's Data handle time (ms):

  time_dataprocessing = {time_dataprocessing}
  time_hc_calculate = {time_hc_calculate}

[end data]
        ''')
    
    mdebug('Output total times (ms):')
    mean_time_dataprocessing = mean(time_dataprocessing)
    mean_time_hc_calculate = mean(time_hc_calculate)
    mean_both = mean_time_dataprocessing + mean_time_hc_calculate
    for key in classifier_names:
        y_lt_totaltime.append((mean(time_literature[key]) + mean_time_dataprocessing) * 1000)
        y_lt_std_error_totaltime.append(sem(time_literature[key]) * 1000)
        
        y_inf_totaltime.append((mean(time_inf_teory[key]) + mean_time_hc_calculate) * 1000)
        y_inf_std_error_totaltime.append(sem(time_inf_teory[key]) * 1000)
        
        y_both_totaltime.append((mean(time_both[key]) + mean_both) * 1000)
        y_both_std_error_totaltime.append(sem(time_both[key]) * 1000)

        mdebug(f'''
    {key}:
      - Literature  : {y_lt_totaltime[-1]} +- {y_lt_std_error_totaltime[-1]}
      - Comp.Entropy: {y_inf_totaltime[-1]} +- {y_inf_std_error_totaltime[-1]}
      - Both        : {y_both_totaltime[-1]} +- {y_both_std_error_totaltime[-1]}
''')

    mdebug(f'''
[data]  Classifier's Total time (ms):

  y_lt_totaltime = {y_lt_totaltime}
  y_lt_std_error_totaltime = {y_lt_std_error_totaltime}
  y_inf_totaltime = {y_inf_totaltime}
  y_inf_std_error_totaltime =  {y_inf_std_error_totaltime}
  y_both_totaltime = {y_both_totaltime}
  y_both_std_error_totaltime = {y_both_std_error_totaltime}

[end data]
        ''')

pltr = PlotResults(classifier_names, SAMPLE_SIZE)
pltr.plot_accuracy(y_lt_accuracy, y_lt_std_error_accuracy,
                y_inf_accuracy, y_inf_std_error_accuracy,
                y_both_accuracy, y_both_std_error_accuracy)
pltr.plot_trainingtime(y_lt_trainingtime, y_lt_std_error_trainingtime,
                    y_inf_trainingtime, y_inf_std_error_trainingtime,
                    y_both_trainingtime, y_both_std_error_trainingtime)
pltr.plot_processingtime(time_dataprocessing, time_hc_calculate)
pltr.plot_totaltime(y_lt_totaltime, y_lt_std_error_totaltime,
                  y_inf_totaltime, y_inf_std_error_totaltime,
                  y_both_totaltime, y_both_std_error_totaltime)
plt.show()