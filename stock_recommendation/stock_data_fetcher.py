from datetime import datetime, timedelta
from enum import Enum
from statistics import stdev
from typing import Callable
# import numpy as np
import pytz
from workalendar.usa import UnitedStates

from stock_recommendation.rate_limited_requester import RateLimitedRequester


class Sorting(Enum):
  """Sorting direction"""
  TOP_N = "descending"
  BOTTOM_N = "ascending"


class Indicator(Enum):
  """Technical indicators"""
  RSI = "rsi"  # Relative strength indicator
  SMA = "sma"  # Simple moving average of prices
  SMAR = "smar"  # Simple moving average of (percentage) returns
  STDR = "stdr"  # Standard deviation of return


class Allocation:
  """Class that contains a ticker and a percent allocation."""

  def __init__(self, ticker: str, percent_allocation: int):  # pragma: no cover
    self.ticker = ticker
    self.percent_allocation = percent_allocation

  def __str__(self):  # pragma: no cover
    """Used when print() is called on the object"""
    return f"{self.ticker}: {self.percent_allocation}"  

  def __repr__(self):  # pragma: no cover
    """Method returns a string representation of the object, useful for debugging."""
    return f"Allocation(ticker={self.ticker}, percent_allocation={self.percent_allocation})"

  @staticmethod
  def create_equal_allocations(tickers: list[str]) -> list['Allocation']:
    if not tickers:
      return [Allocation("No tickers", 100)]

    percent_allocation = 100 / len(tickers)
    allocations = [Allocation(ticker, percent_allocation) for ticker in tickers]
    return allocations


class StockRecommendation:
  """Class that contains a list of stock tickers and and a message on the recommendation."""

  def __init__(self, tickers: list[Allocation], message: str):  # pragma: no cover
    self.tickers = tickers
    self.message = message

  def __str__(self):  # pragma: no cover
    """Used when print() is called on the object"""
    tickers_str = ", ".join(str(allocation) for allocation in self.tickers)
    return f"Tickers: {tickers_str}\nExplanation: {self.message}"

  def __repr__(self):  # pragma: no cover
    """Method returns a string representation of the object, useful for debugging."""
    return f"StockRecommendation(tickers={self.tickers}, message={self.message})"


class StockDataFetcher:
  """Fetches technical indicators for stocks."""

  # Use a list of multiple API keys in order to make more requests at a time. If you only
  # have one, just give a list of one.
  def __init__(self, requester: RateLimitedRequester, twelvedata_api_keys: list[str]):
    self.requester = requester
    self.twelvedata_api_keys = twelvedata_api_keys
    self.current_key_index = 0

  def _get_next_api_key(self):
    """Rotates to the next API key."""
    key = self.twelvedata_api_keys[self.current_key_index]
    # Update the index to the next key, wrapping around to the start if at the end
    self.current_key_index = (self.current_key_index + 1) % len(self.twelvedata_api_keys)
    return key

  def cum_return_pct(self, ticker: str, num_days: int) -> float:  # pragma: no cover
    """
    Returns cumulative return of a ticker as a percent, e.g. cum_return("SPY", 60)
    will return what percent change today's price is from that price 60 days ago.
    """
    today_price = self.current_price(ticker)

    days_ago_date = (datetime.now() - timedelta(days=num_days)).strftime('%Y-%m-%d')
    response = self.requester.get(
        f"https://api.twelvedata.com/eod?apikey={self._get_next_api_key()}&symbol={ticker}&date={days_ago_date}")
    data = response.json()
    previous_price = float(data["close"])

    return (today_price / previous_price - 1) * 100

  def current_price(self, ticker: str) -> float:  # pragma: no cover
    """Returns current price of a stock ticker."""
    response = self.requester.get(
        f"https://api.twelvedata.com/price?apikey={self._get_next_api_key()}&symbol={ticker}")
    data = response.json()
    return float(data["price"])

  def rsi(self, ticker: str, period: int, api_for_current_rsi: bool = True) -> float:  # pragma: no cover
    """
    Returns current RSI (days for timeframe) of a stock ticker.
    If outside of trading hours, use API call to get the latest Relative Strength Index (RSI).
    Otherwise, we need to calculate. To do it for a given ticker and period, e.g. rsi("QQQ", 10)
    will give 10d RSI for QQQ. The RSI calculation requires having a long enough lookback in order
    to be accurate. We use 1000 day lookback here (see API call). Let's say you want a 3 day RSI.
    We will get 1000 days of closing prices first. Then we look at the first 3 days and calculate 
    the RSI. After we do that, we look at the next 3 days (and onwards) using a smoothing function
    which places a stronger emphasis on the most recent gain/loss. You can see this builds on top of
    the previous RSI, which is why having a longer lookback period will make it more accurate, and 
    1000 is good enough.
    Reference: https://www.composer.trade/learn/what-is-relative-strength-index-rsi

    Parameters:
        ticker (str): Ticker, e.g. "QQQ".
        period (int): Time period, e.g. 10 if you want 10d RSI.
        api_for_current_rsi (bool): True will use API to fetch RSIs, which seems to give current
          RSI even when market hasn't closed.

    Returns:
        float: The current RSI.
    """
    # Depend on API to get current RSI
    if api_for_current_rsi:
      response = self.requester.get(
          f"https://api.twelvedata.com/rsi?apikey={self._get_next_api_key()}&symbol={ticker}&time_period={period}&interval=1day&outputsize=10")
      data = response.json()
      current_rsi = data['values'][0]['rsi']
      return float(current_rsi)

    print("Using closing prices to calculate RSI")
    eastern_tz = pytz.timezone('America/New_York')
    current_time_eastern = datetime.now(eastern_tz)
    market_open = current_time_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time_eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    calendar = UnitedStates()
    is_business_day = calendar.is_working_day(current_time_eastern.date())
    is_trading_hours = market_open <= current_time_eastern <= market_close

    # If not during trading hours, just use API to get latest RSI.
    if not is_business_day or not is_trading_hours:
      response = self.requester.get(
          f"https://api.twelvedata.com/rsi?apikey={self._get_next_api_key()}&symbol={ticker}&time_period={period}&interval=1day&outputsize=10")
      data = response.json()
      current_rsi = data['values'][0]['rsi']
      return float(current_rsi)

    # Otherwise, need to calculate RSI ourselves using current price to get an approximation of today's RSI.

    # Get closing prices up to today (last element would be yesterday's closing price)
    today_before_market_open_str = current_time_eastern.replace(
        hour=9, minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
    response = self.requester.get(
        f"https://api.twelvedata.com/time_series?apikey={self._get_next_api_key()}&symbol={ticker}&interval=1day&outputsize=999&end_date={today_before_market_open_str}&order=ASC")
    data = response.json()
    prices = [float(item['close']) for item in data['values']]

    # Append current price to the other prices
    prices.append(self.current_price(ticker))

    # Calculate daily price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))] # changes = np.diff(prices)

    # Initialize lists to hold gains and losses
    gains = [max(change, 0) for change in changes]  # gains = np.maximum(changes, 0)
    losses = [abs(min(change, 0)) for change in changes] # losses = np.abs(np.minimum(changes, 0))

    # Calculate the initial average gain and loss over the first period
    avg_gain = sum(gains[:period]) / period  # avg_gain = np.mean(gains[:period])
    avg_loss = sum(losses[:period]) / period  # avg_loss = np.mean(losses[:period])

    # Store the initial RSI values (none for first `period` days)
    rsi_values = [None] * period # rsi_values = [np.nan] * period  # Initial period will not have RSI values

    # Calculate RSI for the first complete period
    if avg_loss == 0:
      rsi_values.append(100)  # RSI is 100 if there are no losses
    else:
      rs = avg_gain / avg_loss
      rsi_values.append(100 - (100 / (1 + rs)))

    # Calculate RSI for the subsequent periods using smoothing
    for i in range(period, len(gains)):
      current_gain = gains[i]
      current_loss = losses[i]

      # Smooth the average gain and average loss
      avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
      avg_loss = ((avg_loss * (period - 1)) + current_loss) / period

      # Compute RSI
      if avg_loss == 0:
        rsi = 100  # If avg_loss is 0, RSI is 100
      else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

      rsi_values.append(rsi)

    return float(rsi_values[-1])
  
  def ema(self, ticker: str, num_days: int) -> float:  # pragma: no cover
    """
    Returns the <days> day exponential simple moving average of price for the ticker,
    e.g. ema('QQQ', 200) is 200d EMA for QQQ.
    """
    response = self.requester.get(
        f"https://api.twelvedata.com/ema?apikey={self._get_next_api_key()}&symbol={ticker}&time_period={num_days}&interval=1day&outputsize=1")
    data = response.json()
    return float(data["values"][0]["ema"])

  def rsi_lastn(self, ticker: str, period: int, lastn: int) -> float:  # pragma: no cover
    """
    Returns the last N RSI values. For example, if you wanted QQQ's last 20 days of 10d RSI,
    you would call rsi_lastn("QQQ", 10, 20).

    Parameters:
        ticker (str): Ticker, e.g. "QQQ".
        period (int): Time period, e.g. 10 if you want 10d RSI.
        api_for_current_rsi (bool): True will use API to fetch RSIs, which seems to give current
          RSI even when market hasn't closed.
    """
    response = self.requester.get(
        f"https://api.twelvedata.com/rsi?apikey={self._get_next_api_key()}&symbol={ticker}&time_period={period}&interval=1day&outputsize={lastn}")
    data = response.json()
    rsis = [entry['rsi'] for entry in data['values']]
    return rsis

  def sma(self, ticker: str, num_days: int) -> float:  # pragma: no cover
    """
    Returns the <days> day simple moving average of price for the ticker,
    e.g. sma('QQQ', 200) is 200d SMA for QQQ.
    """
    response = self.requester.get(
        f"https://api.twelvedata.com/sma?apikey={self._get_next_api_key()}&symbol={ticker}&time_period={num_days}&interval=1day&outputsize=1")
    data = response.json()
    return float(data["values"][0]["sma"])

  def smar_pct(self, ticker: str, num_days: int) -> float:
    """
    Returns the <days> day simple moving average of percentage returns for the ticker,
    e.g. smar('QQQ', 200) is 200d SMA of retuns for QQQ.
    """
    # Typically there is SMA of prices but not returns. To calculate returns,
    # fetch the days + 1 closing prices, then compute a list of percentages, then
    # get the average of those percentages. For twelvedata API, fetch from Core Data - end of day price
    percentage_returns = self._pct_returns(ticker, num_days, False)

    # Calculate the average of percentage returns
    if percentage_returns:
      return sum(percentage_returns) / len(percentage_returns)

    return 0.0  # Case with no returns

  def sorted_indicator(self, indicator: Indicator, num_days: int, sorting: Sorting, n: int, tickers: list[str]) -> list[str, float]:
    """
    Returns a list of ticker names sorted by to their indicator values, e.g. 
    sorted_indicator(Indicator.RSI, 10, Sorting.TOP_N, 2, ["AAPL", "MSFT", "NVDA"]) would get
    the RSI values for AAPL, MSFT, and NVDA, sort them descending, and then return the top 2.
    """
    # Map the Indicator enum to corresponding functions
    indicator_functions: dict[Indicator, Callable[[str, int], float]] = {
        Indicator.RSI: self.rsi,
        Indicator.SMA: self.sma,
        Indicator.SMAR: self.smar_pct,
        Indicator.STDR: self.std_deviation_return_pct,
    }

    # Get the function corresponding to the indicator enum
    if indicator not in indicator_functions:
      raise ValueError(f"Indicator {indicator} is not supported.")

    indicator_func = indicator_functions[indicator]

    # Calculate indicator values for each ticker
    indicator_values = {}
    for ticker in tickers:
      indicator_values[ticker] = indicator_func(ticker, num_days)

    # Sort indicator values
    sorted_indicator_values = sorted(
        indicator_values.items(),  # This is a list of tuples
        key=lambda item: item[1],
        reverse=(sorting == Sorting.TOP_N)
    )

    # Return the top n
    return sorted_indicator_values[:n]

  def std_deviation_return_pct(self, ticker: str, num_days: int) -> float:
    """The standard deviation of return over the past days."""

    # Fetch percentage returns
    percentage_returns = self._pct_returns(ticker, num_days, False)

    # Calculate standard deviation of percentage returns
    if len(percentage_returns) > 1:  # Need at least two returns to calculate standard deviation
      return stdev(percentage_returns)

    return 0.0  # Case of one or no returns

  def _pct_returns(self, ticker: str, num_days: int, round_decimal: bool) -> list[float]:
    """Returns days of percentage returns for the stock. round_decimal parameter used for testing."""
    today_date = datetime.now().strftime('%Y-%m-%d')
    days_plus_one_ago = (datetime.now() - timedelta(days=num_days + 1)).strftime('%Y-%m-%d')

    response = self.requester.get(
        f"https://api.twelvedata.com/time_series?apikey={self._get_next_api_key()}&symbol={ticker}&interval=1day&start_date={days_plus_one_ago}&end_date={today_date}")
    data = response.json()
    prices = data["values"]  # Includes open, high, low, close, and volume.

    # Extract closing prices from the response
    closing_prices = [float(price["close"]) for price in reversed(prices)]

    # Calculate percentage returns
    percentage_returns = []
    for i in range(1, len(closing_prices)):
      prev_price = closing_prices[i - 1]
      curr_price = closing_prices[i]
      if prev_price != 0:  # Avoid division by zero
        percentage_return = (curr_price - prev_price) / prev_price * 100
        percentage_return = round(percentage_return, 4) if round_decimal else percentage_return
        percentage_returns.append(percentage_return)

    return percentage_returns
