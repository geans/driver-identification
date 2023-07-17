from datetime import timedelta
from sklearn import tree
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import ordpy
import pandas as pd
import time


def get_sublists(original_list, delta):
    pivot = 0
    sublists = []
    len_list = len(original_list)
    shift = 1
    while pivot+delta <= len_list:
        sublists.append(original_list[pivot:pivot+delta])
        pivot += shift
    return sublists

def get_information_binary_label(df_, dx=4, dy=1, taux=1, tauy=1, both=False, LABEL='label', DRIVER='driver', hc_size=30):
    y_label = []
    X = []
    new_df_list = []
    for df, driver in zip([df_[df_[DRIVER] == 'A'], df_[df_[DRIVER] == 'B'],
                          df_[df_[DRIVER] == 'C'], df_[df_[DRIVER] == 'D']],
                         ['A', 'B', 'C', 'D']):
        sliding_window_df = get_sublists(df, hc_size)
        new_df = None
        new_df_sz = 0
        for window_df in sliding_window_df:
            row = {}
            for feature in window_df.drop([LABEL, DRIVER], axis=1).columns:
                h, c = ordpy.complexity_entropy(window_df[feature], dx=dx, dy=dy, taux=taux, tauy=tauy)
                f, s = ordpy.fisher_shannon(window_df[feature], dx=4)
                row[f'{feature}_entropy'] = h
                row[f'{feature}_complexity'] = c
                row[f'{feature}_fisher'] = f
                row[f'{feature}_shannon'] = s
                if both:
                    if window_df[feature].dtypes == int:
                        row[feature] = window_df[feature].values[0]
                    else:
                        row[feature] = window_df[feature].values[0]
            row[LABEL] = window_df[LABEL].values[0]
            row[DRIVER] = driver
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                new_df.loc[new_df_sz] = row
            new_df_sz += 1
        new_df_list.append(new_df)
    return pd.concat(new_df_list)

def get_information_by_driver(df_, dx=4, dy=1, taux=1, tauy=1, both=False, LABEL='label', hc_size=30):
    y_label = []
    X = []
    new_df_list = []
    for df, driver in zip([df_[df_[DRIVER] == 'A'], df_[df_[DRIVER] == 'B'],
                          df_[df_[DRIVER] == 'C'], df_[df_[DRIVER] == 'D']],
                         ['A', 'B', 'C', 'D']):
        sliding_window_df = get_sublists(df, hc_size)
        new_df = None
        new_df_sz = 0
        for window_df in sliding_window_df:
            row = {}
            for feature in window_df.drop([LABEL, DRIVER], axis=1).columns:
                h, c = ordpy.complexity_entropy(window_df[feature], dx=dx, dy=dy, taux=taux, tauy=tauy)
                f, s = ordpy.fisher_shannon(window_df[feature], dx=4)
                row[f'{feature}_entropy'] = h
                row[f'{feature}_complexity'] = c
                row[f'{feature}_fisher'] = f
                row[f'{feature}_shannon'] = s
                if both:
                    if window_df[feature].dtypes == int:
                        row[feature] = int(window_df[feature].head(1))
                    else:
                        row[feature] = float(window_df[feature].head(1))
            row[LABEL] = driver
            row[DRIVER] = driver
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                # new_df.loc[len(new_df)] = row
                new_df.loc[new_df_sz] = row
            new_df_sz += 1
        new_df_list.append(new_df)
    return pd.concat(new_df_list)


if __name__ == '__main__':
    program_time = time.time()
    sz = 50
    label = 'label'
    driver = 'driver'
    binary_class = True

    dfa = pd.read_csv('../../ThisCarIsMine/A/All_1.csv')[:sz]
    dfb = pd.read_csv('../../ThisCarIsMine/B/All_1.csv')[:sz]
    dfc = pd.read_csv('../../ThisCarIsMine/C/All_1.csv')[:sz]
    dfd = pd.read_csv('../../ThisCarIsMine/D/All_1.csv')[:sz]
    dfa[driver] = ['A']*dfa.shape[0]
    dfb[driver] = ['B']*dfb.shape[0]
    dfc[driver] = ['C']*dfc.shape[0]
    dfd[driver] = ['D']*dfd.shape[0]
    dfa[label] = [1]*dfa.shape[0]
    dfb[label] = [0]*dfb.shape[0]
    dfc[label] = [0]*dfc.shape[0]
    dfd[label] = [0]*dfd.shape[0]


    df = pd.concat( [dfa, dfb, dfc, dfd] )
    # print(df.info())
    if binary_class:
        df = get_information_binary_label(df, both=True)
    else:
        df = get_information_by_driver(df, both=True)
    X, y = df.drop([label, driver], axis=1), df[label]

    clf = DecisionTreeClassifier(max_depth=51)
    result_dict = cross_validate(clf, X, y, cv=5, return_estimator=True, return_train_score=True)
    # score = cross_val_score(clf, X, y, cv=5)
    estimator = result_dict['estimator']


    # for i, f in enumerate(X.columns):
    #     print(i, f)

    print('Accuracy:', result_dict['test_score'])
    # print('Accuracy:', score)

    if binary_class:
        out_path = 'plot-tree/binary'
    else:
        out_path = 'plot-tree/by-driver'

    for i, est in enumerate(estimator):
        plt.figure(figsize=(14,10))
        tree.plot_tree(est, feature_names=X.columns, class_names=['driver is not A', 'driver is A'])#, fontsize=20)
        plt.savefig(f'{out_path}/plot-tree-{sz}__{i}.png')


    delta = str(timedelta(seconds=time.time() - program_time))
    print('\n[time]', delta)

    # plt.show()
