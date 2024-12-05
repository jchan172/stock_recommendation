import datetime
import json
import os
import warnings
import numpy as np
import pandas as pd
import quantstats_lumi as qs
import vectorbt as vbt

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


class Backtester:
  """
  Class that can run backtests using VectorBT and generate tear sheet using QuantStats.
  """

  def run_backtest(self,
                   stock_list: list[str],
                   algo: callable,
                   start_date: str,
                   end_date: str,
                   debug: bool = False) -> vbt.Portfolio:
    """
    Runs backtest using VectorBT.

    Parameters:
    - stock_list: A list of tickers without duplicates, such as ['SPY', 'TECL', 'XLK', 'BIL'].
      need to have corresponding JSON files fetched from testfol.io (see the _import_data() method
      for more information).
    - algo: The algorithm function.
    - start_date: A date string in the format YYYY-MM-DD, such as '2016-02-01'
    - end_date: A date string in the format YYYY-MM-DD, such as '2016-02-01'
    - debug: An optional parameter that will print backtest data if True is passed in.

    Returns:
    - A VectorBT Portfolio object.
    """
    print(f"Running backtest for {algo.__name__}...")
    stock_list.sort()
    bt_range = self._import_data(stock_list, start_date, end_date)
    if debug:
      print("Backtest data:")
      print(bt_range)
    allocations, to_trade = self._generate_trades(bt_range, stock_list, algo)

    indicator = vbt.IndicatorFactory(
        class_name="strategy",
        short_name="strategy",
        input_names=["close"],
        param_names=["to_trade"],
        output_names=["signal"]
    ).from_apply_func(
        self._strategy,
        to_trade=to_trade
    )

    results = indicator.run(bt_range)
    entries = results.signal == 1.0
    exits = results.signal == -1.0

    portfolio = vbt.Portfolio.from_signals(bt_range,
                                           entries,
                                           exits,
                                           freq="d",
                                           size=allocations,
                                           size_type='percent',
                                           group_by=True,
                                           cash_sharing=True,
                                           slippage=0.0005,
                                           call_seq='auto',
                                           allow_partial=True,
                                           init_cash=10000)
    return portfolio

  def gen_tear_sheet(self,
                     portfolio: vbt.Portfolio,
                     benchmark: vbt.Portfolio,
                     output_filename: str,
                     debug: bool = False) -> None:
    """
    Generates a QuantStats report, which is a tear sheet showing returns.
    It uses quantstats_lumi because quantstats has a CAGR bug. Note that the benchmark shows up
    as "group", and the strategy under test is called "Strategy". Not sure how to change this.

    Parameters:
    - portfolio: A VectorBT portfolio object
    - benchmark: A VectorBT portfolio object
    - output_filename: A string like "/Users/foo/result.html"
      that is a full path of the HTML file including extension.

    Returns:
    - Nothing. It will create an HTML file saved to location specified by output_filename.
    """
    print("Generating QuantStats tear sheet...")
    returns = portfolio.returns()
    benchmark_returns = benchmark.returns()

    # Ensure the index is datetime and fix data issues
    returns.index = pd.to_datetime(returns.index)
    returns = returns.dropna()  # Drop NaN values
    returns = returns[returns != 0]  # Remove zero returns (if they cause issues)
    returns = returns.astype(float)

    if debug:
      # Check the cleaned returns
      print("Cleaned returns:", returns.describe())

      # Check index integrity
      print("Index type:", type(returns.index))
      print("Any duplicates in index:", returns.index.duplicated().sum())
      print("Is index sorted:", returns.index.is_monotonic_increasing)

    # Use quantstats for analysis
    qs.extend_pandas()  # Extend pandas functionality

    # Create tear sheet to save as HTML file.
    qs.reports.html(returns, benchmark_returns, output=output_filename)

  def _import_data(self,
                   stock_tickers: list[str],
                   start_date: str,
                   end_date: str) -> pd.DataFrame:
    """
    Imports and preprocesses stock data for the given tickers, start date, and end date.
    The JSON files referenced are fetched from testfol.io by looking at the response when you
    click "Backtest" button. Note that you'll be limited to start date ticker with earliest 
    data, e.g. if TECL's earliest start date is 8/17/2008, that will be the earliest you can start 
    even if you have older data for other tickers. You'll need to make sure the data all starts
    and ends at the same time, even if you're going to run different backtests.

    Parameters:
    - start_date: The start date of the backtest range as a string in 'YYYY-MM-DD' format.
    - end_date: The end date of the backtest range as a string in 'YYYY-MM-DD' format.
    - stock_tickers: A list of stock tickers to import data for.

    Returns:
    - A DataFrame containing the preprocessed backtest data, indexed by date.
    """
    data_dir = "stock_data/"  # Directory containing the JSON files
    all_data = pd.DataFrame()
    dates = []

    # Iterate over stock tickers to load JSON data
    for ticker in stock_tickers:
      # Construct file path and assume file names are lowercase
      file_name = os.path.join(data_dir, f"{ticker.lower()}.json")
      try:
        with open(file_name, 'r', encoding='utf-8') as file:
          data = json.load(file)

        prices = data['charts']['history'][1]
        timestamps = data['charts']['history'][0]

        # Parse timestamps into dates
        ticker_dates = [datetime.datetime.fromtimestamp(
            ts).date() + datetime.timedelta(days=1) for ts in timestamps]

        if not dates:
          # Initialize the `dates` list on the first iteration
          dates = ticker_dates

        # Add stock prices to the DataFrame
        all_data[ticker] = prices
      except FileNotFoundError:
        print(f"File for {ticker} not found. Skipping.")
      except KeyError:
        print(f"Unexpected JSON structure in {file_name}. Skipping.")

    # Set the date index
    all_data['timestamp'] = dates
    all_data.set_index('timestamp', inplace=True)

    # Filter data by date range
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()

    bt_range = all_data[(all_data.index >= start) & (all_data.index <= end)]

    return bt_range

  def _generate_trades(self,
                       bt_range:
                       pd.DataFrame,
                       stock_tickers: list,
                       algo_func: callable) -> tuple[pd.Series, pd.DataFrame]:
    """
    Generate trading decisions and allocations based on a provided algorithm.

    Parameters:
    - bt_range: pd.DataFrame containing historical stock prices.
    - stock_tickers: List of stock tickers being considered.
    - algo_func: Callable algorithm function that takes historical data as input and returns 
      a list of selected stock tickers.

    Returns:
    - allocations: pd.Series of portfolio allocations over time.
    - to_trade: pd.DataFrame of trading signals (1: buy, -1: sell, 0: hold).
    """
    to_trade = pd.DataFrame(columns=stock_tickers, dtype="Int64",
                            index=range(len(bt_range.index))).fillna(0)

    allocations = []  # Keep track of ticker allocations
    last_symbol = list([])  # Keep track of currently held stocks

    for x in range(len(bt_range.index)):
      # Need at least 10 days because we're using 10d RSI in our example algo
      if x < 10:
        allocations.append(0)
        continue

      # Get the data up to the current day
      historical_data = bt_range.iloc[:x+1]

      # Run the provided algorithm to get selected stocks
      picks = algo_func(historical_data)

      # Get the current prices for all stocks
      closes = bt_range.iloc[x]

      # Initialize trading decisions for this day
      decisions = list([])

      # Generate trading signals (buy, sell, hold)
      for symbol in closes.index:
        if symbol in picks:  # Stock was selected by the algo
          if symbol in last_symbol:  # Already holding it
            decisions.append(0)  # Hold
          else:
            decisions.append(1)  # Buy
        else:
          if symbol in last_symbol:  # Not selected, but currently holding it
            decisions.append(-1)  # Sell
          else:
            decisions.append(0)  # Hold

      # Update last held symbols and size
      last_symbol = picks
      allocations.append(1 / len(picks) if picks else 0)  # Equal allocation or 0 if no picks
      to_trade.loc[x] = decisions  # Record the decisions for the day

    return pd.Series(allocations), to_trade

  def _strategy(self, close: np.ndarray, to_trade: pd.DataFrame) -> np.ndarray:
    """
    Generate trading signals based on the trading decisions.

    Parameters:
    - close: A 2D NumPy array of close prices, where each row represents a time step
      and each column represents a stock.
    - to_trade: A DataFrame containing trading decisions (1 for buy, -1 for sell, 0 for hold).

    Returns:
    - A 2D NumPy array of trading signals, with the same shape as `close`.
      Each cell contains the signal for a specific stock at a specific time:
      1 for buy, -1 for sell, 0 for hold.
    """
    signal = np.full(close.shape, np.nan)  # Initialize signal array with NaN
    for x in range(len(close)):
      signal[x, :] = to_trade.iloc[x].values  # Populate signal array with trading decisions

    return signal
