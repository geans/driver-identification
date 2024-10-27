import multiprocessing
import os.path
import time

import ordpy
import pandas as pd
import warnings

# Suprimir o aviso específico
warnings.filterwarnings("ignore", message="Be mindful the correct calculation of Fisher information depends on all possible permutations")


class ExtractInformation:
    def __init__(self, df, path_out, window_length, embedding_dimension=6, number_of_threads=1):
        self.__df = df
        self.__path_out = path_out
        self.__window_length = window_length
        self.__d = embedding_dimension
        self.__threads_number = number_of_threads

    def get_parameters(self):
        return self.__d
    
    def run(self):
        threads_list = []
        sublist = self.__getsublist(self.__df, self.__window_length)

        # Separa o Dataframe "sublist" em "self.__threads_number" sublistas de mesmo tamanho para que cada thread trabalhe em uma parte da lista
        sublist_len = len(sublist)
        x = sublist_len // self.__threads_number
        sublist_split = [sublist[i:i+x] for i in range(0, sublist_len, x)]
        
        for i, thread_df in enumerate(sublist_split):
            filename = f'{self.__path_out}.part{i}.csv'
            p = multiprocessing.Process(target=self.__run,
                                        args=(thread_df, filename, self.__d))
            p.start()
            threads_list.append(p)
        return threads_list

    @staticmethod
    def __run(df_list, fileout, dx):
        new_df = None
        new_df_sz = 0
        time_hc = []
        time_fs = []
        for window_df in df_list:
            row = {}
            for feature in window_df.columns:
                window = window_df[feature]
                # window_without_duplicate = window.loc[window.shift() != window]
                if len(window) < dx:
                    h = c = f = s = 'NaN'
                else:
                    t0 = time.time()
                    data_probs = ordpy.ordinal_distribution(window, dx=dx, return_missing=True)[1]
                    tf = time.time()
                    time_probs = tf - t0
                    
                    t0 = time.time()
                    h, c = ordpy.complexity_entropy(data=data_probs, dx=dx, probs=True)
                    tf = time.time()
                    time_hc.append(tf - t0 + time_probs)
                    #
                    t0 = time.time()
                    f, s = ordpy.fisher_shannon(data=data_probs, dx=dx, probs=True)
                    tf = time.time()
                    time_fs.append(tf - t0 + time_probs)
                row[feature] = window_df[feature].values[-1]
                row[f'{feature}_entropy'] = h
                row[f'{feature}_complexity'] = c
                row[f'{feature}_fisher'] = f
                # row[f'{feature}_shannon'] = s
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                new_df.loc[new_df_sz] = row
            new_df_sz += 1
        new_df.to_csv(fileout, index=False)
        time_dict = {
            'time_hc': time_hc,
            'time_fs': time_fs
        }
        time_df = pd.DataFrame.from_dict(time_dict)
        time_df.to_csv(fileout + '.time', index=False)

    @staticmethod
    def __getsublist(original_list, delta):
        pivot = 0
        sublist = []
        list_len = len(original_list)
        shift = 1
        while pivot + delta <= list_len:
            sublist.append(original_list[pivot:pivot + delta])
            pivot += shift
        return sublist


def choose_embedded_dimension(series_length):
    if series_length <= 120:
        return 5
    if series_length <= 720:
        return 6
    if series_length <= 5040:
        return 7
    return 8


def join_parts(number_of_threads_per_file, path_out, path_out_time):
    df_list = []
    df_list_time = []
    for i in range(number_of_threads_per_file):
        filename = f'{path_out}.part{i}.csv'
        if os.path.exists(filename):
            df_list.append(pd.read_csv(filename))
        filename_time = f'{path_out}.part{i}.csv.time'
        if os.path.exists(filename_time):
            df_list_time.append(pd.read_csv(filename_time))
    df = pd.concat(df_list)
    df.to_csv(path_out, index=False)
    df_time = pd.concat(df_list_time)
    df_time.to_csv(path_out_time, index=False)


def dataset__this_car_is_mine():
    print('[dataset] This Car Is Mine!')
    number_of_threads_per_file = 1
    for driver_list, file_amount_list in [('A', 8), ('B', 8), ('C', 5), ('D', 9)]:
        for driver in driver_list:
            # Extract information to parts
#            thread_list = []
#            for series_length in range(60, 901, 60):
#                for file_number in range(1,file_amount_list+1):
#                    d = choose_embedded_dimension(series_length)
#                    directory_in =  f'./datasets/ThisCarIsMine/{driver}'
#                    directory_out = f'./datasets/ThisCarIsMineInf_window{series_length}_dx{d}/{driver}'
#
#                    path_in = f'{directory_in}/All_{file_number}.csv'
#                    path_out =f'{directory_out}/All_{file_number}.csv'
#
#                    print(path_out)
#                    if not os.path.exists(directory_out):
#                        print('\t[create path]', directory_out)
#                        os.makedirs(directory_out, exist_ok=True)
#
#                    df = pd.read_csv(path_in)
#                    thread_list += ExtractInformation(
#                        df=df, 
#                        path_out=path_out, 
#                        window_length=series_length, 
#                        embedding_dimension=d, 
#                        number_of_threads=number_of_threads_per_file
#                    ).run()
#            print('Total', len(thread_list), 'threads')
#            for i, thread in enumerate(thread_list, start=1):
#                print(f'{i}/{len(thread_list)}')
#                thread.join()
            
            # Join parts to one csv file
            for file_number in range(1,file_amount_list+1):
                thread_list = []
                for series_length in range(60, 901, 60):
                    d = choose_embedded_dimension(series_length)
                    directory_out = f'./datasets/ThisCarIsMineInf_window{series_length}_dx{d}/{driver}'
                    path_out =f'{directory_out}/All_{file_number}.csv'
                    path_out_time =f'{directory_out}/All_{file_number}.csv.time'
                    print(path_out)
                    if not os.path.exists(path_out) or not os.path.exists(path_out_time):
                        p = multiprocessing.Process(target=join_parts, 
                                                    args=(number_of_threads_per_file, 
                                                          path_out, path_out_time))
                        p.start()
                        thread_list.append(p)
                print('Total', len(thread_list), 'threads')
                for i, thread in enumerate(thread_list, start=1):
                    print(f'{i}/{len(thread_list)}')
                    thread.join()



if __name__ == '__main__':
    dataset__this_car_is_mine()
