from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf


def my_debug(*objects, sep=' ', end='\n', file=None, flush=False, path='.'):
    if False:
        print(*objects, sep=sep, end=end, file=file, flush=flush)
    # output_file = open(f'{path}/dlc_log.txt', 'a')
    # for obj in objects:
    #     output_file.write(str(obj))
    #     output_file.write(sep)
    # output_file.write(end)
    # output_file.close()

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=5, batch_size=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.time_steps = 1

    def __model_builder(self, input_dim):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(160, input_dim=input_dim, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(120, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

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
    
    def fit(self, X, y):
        # X_numpy = X.to_numpy()
        # if len(X_numpy.shape) == 2:
        #     X_reshaped = np.reshape(X_numpy, (X_numpy.shape[0] // self.time_steps, self.time_steps, X_numpy.shape[1]))
        # input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])
        # self.model = self.__model_builder(input_shape)
        # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # self.model.fit(X_reshaped, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[early_stopping], validation_split=0.2)
        # return self
        self.model = self.__model_builder(X.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        with tf.device('/GPU:0'):
            self.model.fit(X,y, epochs=10,batch_size=100, shuffle=False)
        return self
    
    def predict(self, X):
        # X_reshaped = X.to_numpy()
        # if len(X_reshaped.shape) == 2:
        #     X_reshaped = np.reshape(X_reshaped, (X_reshaped.shape[0], self.time_steps, X_reshaped.shape[1] // self.time_steps))
        # y_pred = self.model.predict(X_reshaped)
        # y_pred = (y_pred >= 0.5).astype(int)
        y_pred = self.model.predict(X)
        y_pred = (y_pred >= 0.5).astype(int)
        return y_pred
