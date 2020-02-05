from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from getdata import *

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

def start_process_machine(ticker):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #For, the number of days is 100 (69 business days), but adjust in the getdata.py file

    get_historical_data(ticker)

    dates, prices = data_to_array(ticker)

    dates = np.array(dates)
    prices = np.array(prices)

    df = pd.DataFrame({'Date': dates, 'Price': prices})

    uni_data = df['Price']
    uni_data.index = df['Date']

    uni_data = uni_data.values

    TRAIN_SPLIT = round(len(uni_data) * 9 / 10)
    tf.random.set_seed(13)

    uni_data_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_data_std = uni_data[:TRAIN_SPLIT].std()

    std = uni_data.std()
    mean = uni_data.mean()

    uni_data = (uni_data - uni_data_mean) / uni_data_std

    #how many days are given after the train split
    univariate_past_history = round(len(uni_data) * 1 / 10) - 1
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    '''print(x_val_uni, y_val_uni)

    print('Single window of past history')
    print(x_train_uni[0])
    print('\n Target temperature to predict')
    print(y_train_uni[0])'''

    #LSTM
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')

    EVALUATION_INTERVAL = 200
    EPOCHS = 10

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)

    for x, y in val_univariate.take(3):
        predict = simple_lstm_model.predict(x)[0]
        predict = (predict*std) + mean

    return predict

    #show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
    #          'Baseline Prediction Example')


def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
      labels = ['History', 'True Future', 'Model Prediction']
      marker = ['.-', 'rx', 'go']
      time_steps = create_time_steps(plot_data[0].shape[0])
      if delta:
        future = delta
      else:
        future = 0

      plt.title(title)
      for i, x in enumerate(plot_data):
        if i:
          plt.plot(future, plot_data[i], marker[i], markersize=10,
                   label=labels[i])
        else:
          plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
      plt.legend()
      plt.xlim([time_steps[0], (future+5)*2])
      plt.xlabel('Time-Step')
      plt.show()
      return plt

def baseline(history):
  return np.mean(history)


