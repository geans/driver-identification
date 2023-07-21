from datetime import timedelta
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import time

from getdata import GetData
from information import InformationHandle


if __name__ == '__main__':
    program_time = time.time()
    sz = 50
    binary_class = True
    label = 'label'
    driver = 'driver'
    inf_handle = InformationHandle(label, driver)
    data_handle = GetData('../../ThisCarIsMine', label, driver)

    df = data_handle.get_sample(sz, driver_target='A')
    if binary_class:
        df = inf_handle.get_information_binary_label(df, both=True)
    else:
        df = inf_handle.get_information_by_driver(df, both=True)
    X, y = df.drop([label, driver], axis=1), df[label]

    clf = DecisionTreeClassifier(max_depth=51)
    result_dict = cross_validate(clf, X, y, cv=5, return_estimator=True, return_train_score=True)
    estimator = result_dict['estimator']

    print('Accuracy:', result_dict['test_score'])

    if binary_class:
        out_path = 'results/plot-tree/binary'
    else:
        out_path = 'results/plot-tree/by-driver'

    for i, est in enumerate(estimator):
        plt.figure(figsize=(14, 10))
        tree.plot_tree(est, feature_names=X.columns, class_names=['driver is not A', 'driver is A'])
        plt.savefig(f'{out_path}/plot-tree-{sz}__{i}.png')

    delta = str(timedelta(seconds=time.time() - program_time))
    print('\n[time]', delta)

    # plt.show()
