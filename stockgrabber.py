#getdata
from getdata import *

#regression.py
from regression import *

#sendmessage.py
from sendmessage import *

#timeseries.py
from timeseries import *

#timeseries2.py
from timeseries2 import *

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

'''ticker = 'AAPL'

get_historical_data(ticker)
dates, prices = data_to_array(ticker)
linear_projection = simple_regression(dates, prices, ticker)
machine_projection = start_process_machine(ticker)

print(ticker, ": ", machine_projection, linear_projection)
if machine_projection or linear_projection > prices[len(prices) - 1]:
    print("Profitable")'''

profit_list = []

ticker = "AAPL"

run_model(ticker)
















