import pandas as pd
from sklearn import preprocessing

import pandas_datareader as pdr

import datetime

import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers

import numpy as np

np.random.seed(4)

tf.random.set_seed(4)

history_points = 50

#Save Data

from alpha_vantage.timeseries import TimeSeries
import json


def save_dataset(ticker):
    api_key = 'EX7ISCFA4F846V1E'

    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(ticker, outputsize='full')

    data.to_csv('/Users/akim0417/stock/csv/' + ticker + '.csv')

def reverse_array(array):
    return [ele for ele in reversed(array)]


def create_data(ticker):
    save_dataset(ticker)

    data = pd.read_csv('/Users/akim0417/stock/csv/' + ticker + '.csv')
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)

    data = data.values

    '''
    
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    open = data['Open'].values
    high  = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values

    open, high, low, close, volume = reverse_array(open), reverse_array(high), reverse_array(low), reverse_array(close), reverse_array(volume)

    data = pd.DataFrame({
        'Open': open,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    '''

    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    ohlcv_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][j + history_points].copy() for j in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:, 3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]

    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # note since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:, 3])
        macd = calc_ema(his, 12) - calc_ema(his, 26)
        technical_indicators.append(np.array([sma]))
        # technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == \
           technical_indicators_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser

def run_model(ticker):
    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = create_data(ticker)

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]

    print(ohlcv_train.shape)
    print(ohlcv_test.shape)

    # model architecture

    # define two sets of inputs
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')
    dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)

    # evaluation

    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict([ohlcv_histories, technical_indicators])
    y_predicted = y_normaliser.inverse_transform(y_predicted)
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)

    import matplotlib.pyplot as plt

    plt.gcf().set_size_inches(22, 15, forward=True)

    start = 0
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    # real = plt.plot(unscaled_y[start:end], label='real')
    # pred = plt.plot(y_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])

    plt.show()
