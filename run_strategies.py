from datetime import datetime
import os
import pprint
from dotenv import load_dotenv

from stock_recommendation.rate_limited_requester import RateLimitedRequester
from stock_recommendation.stock_data_fetcher import *
from stock_recommendation.strategies.basic_example_strategy import basic_strategy


# This script meant to be run locally
load_dotenv()
API_KEY = os.getenv('API_KEY')
requester = RateLimitedRequester(rate_limit=8, per=60)
stocks = StockDataFetcher(requester, [API_KEY])

print("Running basic strategy...")
result = basic_strategy(stocks)
print(f"{result.tickers}, {result.message}")

# ticker = "SPY"
# Get RSI from StockDataFetcher
# print(stocks.rsi("SPY", 60))

# Get closing prices from API
# today_date = datetime.now().strftime('%Y-%m-%d')
# time_series = requester.get(
#   f"https://api.twelvedata.com/time_series?apikey={API_KEY}&symbol={ticker}&interval=1day&outputsize=10&end_date={today_date}")
# print("Closing prices from API:")
# pprint.pp(time_series.json())

# Get RSI values from API
# rsis = requester.get(
#     f"https://api.twelvedata.com/rsi?apikey={API_KEY}&symbol={ticker}&time_period=60&interval=1day&outputsize=10")
# print("RSIs from API:")
# data = rsis.json()
# pprint.pp(data['values'][0]['rsi'])
