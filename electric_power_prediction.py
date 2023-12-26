import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.callbacks import Callback
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error


def download_and_extract_data():
    url = 'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()



def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.shuffle(1000)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:-n_past], w[-n_past:, :1]))
    return ds.batch(batch_size).prefetch(1)

class ValidationAccuracyStop(Callback):
    def __init__(self, mae=0.05 ):
        super(ValidationAccuracyStop, self).__init__()
        self.mae = mae

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('mae') <  self.mae:
            print("\nmae of %0.2f%% reached, stopping training." % (self.mae * 100))
            self.model.stop_training = True
download_and_extract_data()
df = pd.read_csv('household_power_consumption.csv', sep=',',
                    infer_datetime_format=True, index_col='datetime', header=0)

N_FEATURES = 7

data = df.values
split_time = int(len(data) * 0.5)
data = normalize_series(data, data.min(axis=0), data.max(axis=0))

x_train = data[:split_time]
x_valid = data[split_time:]

BATCH_SIZE = 32
N_PAST = 24  # Number of past time steps based on which future observations should be predicted
N_FUTURE = 24  # Number of future time steps which are to be predicted.
SHIFT = 1  # By how many positions the window slides to create a new window of observations.

train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                n_past=N_PAST, n_future=N_FUTURE,
                                shift=SHIFT)
valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                n_past=N_PAST, n_future=N_FUTURE,
                                shift=SHIFT)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(256, input_shape=(24, 7)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7),
    tf.keras.layers.Dense(N_FUTURE * N_FEATURES),
    tf.keras.layers.Reshape([N_FUTURE, N_FEATURES])
])

optimizer =tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
model.fit(train_set, validation_data=valid_set,epochs=100, callbacks=[ValidationAccuracyStop()])
