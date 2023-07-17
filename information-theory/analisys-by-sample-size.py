import multiprocessing
import time
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import sem

import config
import datapreprocessing as dpp
from information import InformationHandle
from getdata import GetData

if __name__ == '__main__':
    program_time = time.time()

    sz = 50
    sample_size_list = list(range(40, 201, 10))
    dataserver = GetData(config.path_dataset, config.label, config.driver)

    label = config.label
    driver = config.driver
    driver_list = ['A', 'B', 'C', 'D']
    path = '../../ThisCarIsMine'
    lit = 'lit'
    inf = 'inf'
    value = 'value'
    err = 'err'
    score_by_sample_size = {lit: {value: [], err: []},
                            inf: {value: [], err: []}}
    score_list_literature = []
    score_list_information = []

    running_process = []
    clf = DecisionTreeClassifier(max_depth=51)
    # clf = SVC(gamma=.9, C=1)
    infhandle = InformationHandle(label, driver)


    def classifier_handle(clf, df, technique):
        pp = dpp.LiteraturePreprocessing(df)
        pp.remove_correlation()
        pp.remove_invariants()
        pp.normalization()
        pp.remove_miss_value()
        if technique == inf:
            df = infhandle.get_information_binary_label(pp.get_df(), both=True)
        else:
            pp.window()
            df = pp.get_df()
        X, y = df.drop([label, driver], axis=1), df[label]
        result_dict = cross_validate(clf, X, y,
                                     cv=5, return_estimator=False, return_train_score=False)
        score[technique] += [result_dict['test_score']]


    for sz in sample_size_list:
        manager = multiprocessing.Manager()
        score = manager.dict({lit: [], inf: []})
        print(f'{sz} / {sample_size_list[-1]}')

        for i in range(5):

            # literature
            for d in driver_list:
                df = dataserver.get_sample(samplesize=sz,
                                           driver_target=d)
                p = multiprocessing.Process(target=classifier_handle,
                                            args=(clf, df, lit))
                p.start()
                running_process.append(p)

            # literature + information
            for d in driver_list:
                df = dataserver.get_sample(samplesize=sz,
                                           driver_target=d)
                p = multiprocessing.Process(target=classifier_handle,
                                            args=(clf, df, inf))
                p.start()
                running_process.append(p)

            for p in running_process:
                p.join()
        for tec in [lit, inf]:
            score_by_sample_size[tec][value] += [np.mean(score[tec])]
            score_by_sample_size[tec][err] += [np.mean(sem(score[tec]))]

    print(f'\nsample_size_list = {sample_size_list}')
    print(f'\nscore_by_sample_size[lit] = {score_by_sample_size[lit]}')
    print(f'\nscore_by_sample_size[inf] = {score_by_sample_size[inf]}\n')

    # plot

    bar_width = 0.15
    bottomsize = 0.15
    fontsize = 15
    X_axis = np.arange(len(sample_size_list))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))


    def autolabel(rects, ax, vertical_pos=.5, fontsize=10):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., vertical_pos,
                    f'{height:3.2}', fontsize=fontsize,
                    ha='center', va='bottom')


    bars_lit = ax.bar(X_axis - bar_width * 1.5,
                      score_by_sample_size[lit][value],
                      bar_width * 3,
                      label='Literature',
                      yerr=score_by_sample_size[lit][err])
    bars_inf = ax.bar(X_axis + bar_width * 1.5,
                      score_by_sample_size[inf][value],
                      bar_width * 3,
                      label='Literature + Information Theory',
                      yerr=score_by_sample_size[inf][err])
    ax.set_xticks(X_axis, sample_size_list)
    ax.set_ylim([0.5, 1.05])
    ax.set_xlabel('Sample size', fontsize=fontsize * 1.5)
    ax.set_ylabel('Accuracy', fontsize=fontsize * 1.5)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -bottomsize),
              fancybox=True, shadow=True, ncol=3, fontsize=fontsize)
    fig.subplots_adjust(bottom=bottomsize * 1.5)
    autolabel(bars_lit, ax)
    autolabel(bars_inf, ax)
    plt.savefig(f'accuracy-by-samplesize__{sample_size_list[-1]}.png')

    delta = str(timedelta(seconds=time.time() - program_time))
    print('\n[time]', delta)

    plt.show()
