#webdriver
from selenium import webdriver

#pandas
import pandas_datareader as pdr
import pandas as pd

#csv
import csv

#datetime
import datetime
import datetime as dt

#numpy
import numpy as np

DJIA = ["MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", "DD", "XOM", "GE", "GS", "HD", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UTX", "UNH", "VZ", "V", "WMT"]

aggressive_small_caps = "https://finance.yahoo.com/screener/predefined/aggressive_small_caps"

daily_gainers = "https://finance.yahoo.com/screener/predefined/day_gainers"

growth_technology = "https://finance.yahoo.com/screener/predefined/growth_technology_stocks"

def create_ticker_list(number, url):

    driver = webdriver.Chrome(
        '/usr/local/bin/chromedriver'
    )

    driver.get(url)

    stock_list = []

    number += 1

    for i in range(1, number):
        ticker = driver.find_element_by_xpath(

            '//*[@id="scr-res-table"]/div[1]/table/tbody/tr[' + str(i) + ']/td[1]/a'

        )

        stock_list.append(ticker.text)

    return stock_list

def get_historical_data(ticker):
    startdate = datetime.datetime.now() - datetime.timedelta(days = 100)
    temp = pdr.get_data_yahoo(symbols=ticker, start= startdate, end=datetime.datetime.now())
    save_historical_data(temp['Adj Close'], ticker)

def save_historical_data(data, ticker):

    data.to_csv('/Users/akim0417/stock/csv/' + ticker + '.csv', header = False)

    #adding headers - revisit later
    '''with open('/Users/akim0417/stock/csv/' + ticker + '.csv', 'wb') as f:
        w = csv.writer(f)
        w.writerow(["Date", "Price"])
        for ticker in data:
            w.writerow(ticker)'''

def data_to_array(ticker):
    dates = []
    prices = []

    data = csv.reader(open('/Users/akim0417/stock/csv/' + ticker + '.csv', 'r+'), delimiter = ",", quotechar = '|')

    for row in data:
        dates.append(row[0])
        prices.append(row[1])

    for i in range(0, len(prices)):
        prices[i] = float(prices[i])

    for j in range(0, len(dates)):
        dates[j] = datetime.datetime.strptime(dates[j], '%Y-%m-%d')

    return dates, prices