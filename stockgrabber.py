#numpy
import numpy as np

#getdata
from getdata import *

#for Directories
import os

#for regression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib

#datetime
import datetime
from datetime import timezone

def simple_regression(dates, prices, ticker):

    y = prices
    x1 = [dt.replace(tzinfo=timezone.utc).timestamp() for dt in dates]

    x1 = np.asarray(x1)

    #run regression

    x = sm.add_constant(x1)
    results = sm.OLS(y,x).fit()

    #evaluate coefficients

    df = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]
    coefficient = df['coef'].values[1]
    constant = df['coef'].values[0]

    coefficient = coefficient.astype(type('float', (float,), {}))
    constant = constant.astype(type('float', (float,), {}))

    yhat = compute_line(coefficient, constant, x1)

    #prediction

    tomorrow = datetime.datetime.now() + datetime.timedelta(days = 1)
    tomorrow_utc = tomorrow.replace(tzinfo=timezone.utc).timestamp()

    x2 = [tomorrow.replace(tzinfo=timezone.utc).timestamp()]

    x2 = np.asarray(x2)

    prediction_value = compute_line(coefficient, constant, x2)

    #plot
    if ticker == 'WMT':
        plt.scatter(x1, y)
        fig = plt.plot(x1, yhat, lw=4, c='orange', label = 'regression line')
        projection = plt.plot([tomorrow_utc], [prediction_value], marker='o', markersize=3, color="red")
        plt.xlabel('Date (10^9)', fontsize = 20)
        plt.ylabel('Prices', fontsize = 20)
        plt.xticks(rotation = 45)

        plt.show()

    #print final values
    print(ticker)
    print(prediction_value)

def compute_line(coefficient, constant, x1):
    return(coefficient * x1 + constant)

stock_list = []

#DJIA
stock_list = DJIA

#daily gainers
#stock_list = create_ticker_list(15, daily_gainers)

#aggressive small caps
#stock_list = create_ticker_list(15, aggressive_small_caps)

#growth technology
#stock_list = create_ticker_list(15, growth_technology)

#for ticker in stock_list:
#    get_historical_data(ticker)

tomorrow = datetime.datetime.now() + datetime.timedelta(days = 1)

print(tomorrow)

for ticker in DJIA:
    get_historical_data(ticker)
    dates, prices = data_to_array(ticker)

    simple_regression(dates, prices, ticker)














