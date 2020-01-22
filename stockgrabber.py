#getdata
from getdata import *

#regression.py
from regression import *

#sendmessage.py
from sendmessage import *

#datetime
import datetime

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

order = {}

for ticker in DJIA:
    get_historical_data(ticker)
    dates, prices = data_to_array(ticker)

    margin = simple_regression(dates, prices, ticker)

    order[ticker] = margin

most_profitable = sorted(order.items(), key=lambda x: x[1], reverse=True)
print(most_profitable)














