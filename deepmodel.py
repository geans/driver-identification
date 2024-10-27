import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping
import json
from scipy.stats import sem
# import logging
# logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Make TensorFlow log less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def save_to_json(metrics, name):
    with open(name, 'w') as f:
        json.dump(metrics, f, indent=4)

def evaluate_metrics(y_test, y_pred):
    return {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'roc_auc': metrics.roc_auc_score(y_test, y_pred),
        'precision': metrics.precision_score(y_test, y_pred, zero_division=0),
        'recall': metrics.recall_score(y_test, y_pred)
    }

def model_builder(input_dim, output_dim):
    # model = tf.keras.Sequential()

    # lstm_units = [64, 128, 256]
    # lstm_dropouts = [0.2, 0.3, 0.4]

    # model.add(LSTM(units=6, dropout=0.2, return_sequences=True))

    # for units, dropout in zip(lstm_units, lstm_dropouts):
    #     model.add(LSTM(units, dropout=dropout, return_sequences=True, input_shape=input_shape))

    # model.add(LSTM(128))
    # model.add(Dropout(0.4))
    # model.add(Dense(256))
    # model.add(Dense(1, activation='sigmoid'))

    # learning_rate = 0.001

    # model.compile(
    #     loss=tf.keras.losses.BinaryCrossentropy(),
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    #     metrics=[
    #         tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    #         tf.keras.metrics.Precision(name='precision'),
    #         tf.keras.metrics.Recall(name='recall')
    #     ]
    # )
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(160, input_dim=input_dim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])

    # model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        # metrics=[
        #     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        #     tf.keras.metrics.Precision(name='precision'),
        #     tf.keras.metrics.Recall(name='recall')
        # ]
    )

    return model

def evaluate_lstm_helper(args):
    return evaluate_lstm(*args)

def evaluate_lstm(X_train, X_test, y_train, y_test):

    # time_steps = 5

    # train_samples_length = X_train.shape[0]//time_steps
    # train_features_length = X_train.shape[1]
    # test_samples_length = X_test.shape[0]//time_steps
    # test_features_length = X_test.shape[1]

    # train_array_length = train_samples_length * time_steps
    # test_array_length = test_samples_length * time_steps

    # X_train = X_train[:train_array_length]
    # X_test = X_test[:test_array_length]
    # y_train = y_train[:train_array_length]
    # y_test = y_test[:test_array_length]

    # X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    # X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test
    # X_train = np.reshape(X_train, (train_samples_length, time_steps, train_features_length))
    # X_test = np.reshape(X_test, (test_samples_length, time_steps, test_features_length))
    
    model = model_builder(input_dim=X_train.shape[1], output_dim=1)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=2, callbacks=[early_stopping], validation_split=0.2)
    with tf.device('/GPU:0'):
        model.fit(X_train,y_train, epochs=10,batch_size=100, shuffle=False)

    y_pred = model.predict(X_test)
    y_pred = (y_pred >= 0.5).astype(int)

    # salva y_test e y_pred para um único arquivo
    np.savetxt("y_test.csv", y_test, delimiter=",")
    np.savetxt("y_pred.csv", y_pred, delimiter=",")

    return evaluate_metrics(y_test, y_pred)

def build_scores_arrays(metrics_dict):
    accuracy_mean_list, accuracy_sem_list = [], []
    roc_auc_mean_list, roc_auc_sem_list = [], []
    precision_mean_list, precision_sem_list = [], []
    # for data_name in metrics_dict.keys():
    accuracy_arr = [metrics_dict['accuracy']]
    roc_auc_arr = [metrics_dict['roc_auc']]
    precision_arr = [metrics_dict['precision']]
    accuracy_mean_list.append(np.mean(accuracy_arr))
    accuracy_sem_list.append(sem(accuracy_arr))
    roc_auc_mean_list.append(np.mean(roc_auc_arr))
    roc_auc_sem_list.append(sem(roc_auc_arr))
    precision_mean_list.append(np.mean(precision_arr))
    precision_sem_list.append(sem(precision_arr))
    accuracy_sem_list = 0
    roc_auc_sem_list = 0
    precision_sem_list = 0
    return {
        'accuracy': {'mean': accuracy_mean_list, 'sem': accuracy_sem_list},
        'roc_auc': {'mean': roc_auc_mean_list, 'sem': roc_auc_sem_list},
        'precision': {'mean': precision_mean_list, 'sem': precision_sem_list},
    }

def plot_metrics(metrics_dict, model_name):
    scores_dict = build_scores_arrays(metrics_dict)
    font_size = 24
    plt.figure(figsize=(18,10)) #30 ficou bom
    X_names = metrics_dict.keys()
    X_axis = np.arange(len(X_names))
    plt.bar(X_axis - 0.2, scores_dict['accuracy']['mean'], 0.2,
            yerr=scores_dict['accuracy']['sem'], label='Acurácia', edgecolor="black")
    plt.bar(X_axis, scores_dict['roc_auc']['mean'], 0.2,
            yerr=scores_dict['roc_auc']['sem'], label='ROC AUC', edgecolor="black")
    plt.bar(X_axis + 0.2, scores_dict['precision']['mean'], 0.2,
            yerr=scores_dict['precision']['sem'], label='Precisão', edgecolor="black")
    plt.xticks(X_axis, X_names)
    plt.tick_params(axis='both', which='major', labelsize=font_size)
    plt.ylim((0.5, 1))
    plt.xlabel('Método', fontsize=font_size)
    plt.ylabel('Pontuação', fontsize=font_size)
    plt.legend(fontsize=font_size - 4)
    plt.savefig(f'results/graphic_{model_name}.png')

    
if __name__ == '__main__':
    class_feat = 'driver'
    df_a = pd.read_csv('datasets/ThisCarIsMineInf_window60_dx5/A/All_1.csv', index_col=False)
    df_b = pd.read_csv('datasets/ThisCarIsMineInf_window60_dx5/B/All_1.csv', index_col=False)
    df_a[class_feat] = [1] * df_a.shape[0]
    df_b[class_feat] = [0] * df_b.shape[0]
    df = pd.concat([df_a, df_b])
    X, y = df.drop([class_feat], axis=1), df[class_feat]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    metrics_dict = evaluate_lstm(X_train, X_test, y_train, y_test)

    # salva as métricas em um arquivo JSON
    save_to_json(metrics_dict, 'results/metrics.json')

    plot_metrics(metrics_dict, 'lstm')

