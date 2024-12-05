from enum import Enum
import unittest
from tests.test_utils import parameterized  # Adjust this import according to your directory structure
from unittest.mock import patch, MagicMock

from statistics import stdev
from stock_recommendation.rate_limited_requester import RateLimitedRequester
from stock_recommendation.stock_data_fetcher import Allocation, Indicator, Sorting, StockDataFetcher


class TestStockDataFetcher(unittest.TestCase):
  """Tests stock data fetcher"""

  def setUp(self):
    # Create a mock requester
    self.requester = MagicMock(spec=RateLimitedRequester)
    # Initialize the StockDataFetcher with the mock requester
    self.stock_fetcher = StockDataFetcher(self.requester, "ABC123")

  def test_sorted_indicator(self):
    # Arrange
    tickers = ["AAPL", "MSFT", "NVDA"]
    days = 10
    n = 2
    indicator = Indicator.RSI
    sorting = Sorting.TOP_N

    # Mock the rsi method to return predefined values
    # Mocked values for AAPL, MSFT, NVDA
    self.stock_fetcher.rsi = MagicMock(side_effect=[70.0, 65.0, 80.0])

    # Act
    result = self.stock_fetcher.sorted_indicator(indicator, days, sorting, n, tickers)

    # Assert
    expected_result = [
        ("NVDA", 80.0),
        ("AAPL", 70.0)
    ]
    self.assertEqual(result, expected_result)

    # Check that rsi was called with the correct parameters
    self.stock_fetcher.rsi.assert_any_call("AAPL", days)
    self.stock_fetcher.rsi.assert_any_call("MSFT", days)
    self.stock_fetcher.rsi.assert_any_call("NVDA", days)

  def test_sorted_indicator_unsupported_indicator(self):
    """Test that an unsupported indicator raises a ValueError."""
    tickers = ['AAPL', 'GOOGL', 'TSLA']

    # Creating a mock unsupported Indicator (assuming the Indicator enum doesn't have 'MACD')
    class UnsupportedIndicator(Enum):
      MACD = "macd"

    with self.assertRaises(ValueError) as context:
      self.stock_fetcher.sorted_indicator(UnsupportedIndicator.MACD, num_days=14,
                                          sorting=Sorting.TOP_N, n=2, tickers=tickers)

    # Check if the correct error message is raised
    self.assertEqual(str(context.exception),
                     "Indicator UnsupportedIndicator.MACD is not supported.")

  @patch.object(StockDataFetcher, '_pct_returns')
  def test_smar(self, mock_private_method):
    # Set the return value for the private method inside the test method
    mock_private_method.return_value = [1.0, -2.2, 4.5]

    result = self.stock_fetcher.smar_pct("AAPL", 3)  # Test for 3 days of data

    # Calculate the expected average return:
    expected_average = sum(mock_private_method.return_value) / 3

    # Assert that the result is approximately equal to the expected average
    self.assertAlmostEqual(result, expected_average, places=4)

  @patch.object(StockDataFetcher, '_pct_returns')
  def test_smar_no_prices(self, mock_private_method):
    # Set the return value for the private method inside the test method
    mock_private_method.return_value = []

    result = self.stock_fetcher.smar_pct("AAPL", 3)

    # If there are no prices, the expected return should be 0.0
    self.assertEqual(result, 0.0)

  @patch.object(StockDataFetcher, '_pct_returns')
  def test_std_deviation_return_pct(self, mock_private_method):
    # Set the return value for the private method inside the test method
    mock_private_method.return_value = [1.0, -2.2, 4.5]

    result = self.stock_fetcher.std_deviation_return_pct("AAPL", 3)  # Test for 3 days of data

    # Calculate the expected standard deviation of return:
    expected_std_deviation = stdev(mock_private_method.return_value)

    # Assert that the result is approximately equal to the expected standard deviation
    self.assertAlmostEqual(result, expected_std_deviation, places=4)

  @patch.object(StockDataFetcher, '_pct_returns')
  def test_std_deviation_return_pct_one_return(self, mock_private_method):
    # Set the return value for the private method inside the test method
    mock_private_method.return_value = [1.0]

    result = self.stock_fetcher.std_deviation_return_pct("AAPL", 3)  # Test for 3 days of data

    # Expected standard deviation with no only one percentage return would be 0.
    expected_std_deviation = 0.0

    # Assert that the result is approximately equal to the expected standard deviation
    self.assertEqual(result, expected_std_deviation)

  @patch.object(StockDataFetcher, '_pct_returns')
  def test_std_deviation_return_pct_no_returns(self, mock_private_method):
    # Set the return value for the private method inside the test method
    mock_private_method.return_value = []

    result = self.stock_fetcher.std_deviation_return_pct("AAPL", 3)  # Test for 3 days of data

    # Expected standard deviation with no percentages would be 0.
    expected_std_deviation = 0.0

    # Assert that the result is approximately equal to the expected average
    self.assertEqual(result, expected_std_deviation)

  def test_pct_returns(self):
    # Mock the API response
    mock_response_data = {
        "values": [
            {"close": "120"},  # Most recent
            {"close": "115"},
            {"close": "105"},
            {"close": "110"},
            {"close": "100"},  # Oldest
        ]
    }
    self.requester.get.return_value.json.return_value = mock_response_data

    # Call the _pct_returns method
    result = self.stock_fetcher._pct_returns(ticker='AAPL', num_days=4, round_decimal=True)

    # Expected percentage returns calculation
    expected_returns = [
        round((110 - 100) / 100 * 100, 4),
        round((105 - 110) / 110 * 100, 4),
        round((115 - 105) / 105 * 100, 4),
        round((120 - 115) / 115 * 100, 4),
    ]

    # Assert that the result matches the expected returns
    self.assertEqual(result, expected_returns)


class TestAllocation(unittest.TestCase):
  """Tests the Allocation class's static method in particular."""
  @parameterized(
      (['AAPL', 'GOOGL', 'MSFT'], [{'ticker': 'AAPL', 'percent_allocation': 33.33},
                                   {'ticker': 'GOOGL', 'percent_allocation': 33.33},
                                   {'ticker': 'MSFT', 'percent_allocation': 33.33}]),
      ([], [{'ticker': "No tickers", 'percent_allocation': 100}]),
      (['AAPL'], [{'ticker': 'AAPL', 'percent_allocation': 100}]),
  )
  def test_create_equal_allocations(self, tickers, expected):
    allocations = Allocation.create_equal_allocations(tickers)
    self.assertEqual(len(allocations), len(expected))
    for allocation, exp in zip(allocations, expected):
      self.assertEqual(allocation.ticker, exp['ticker'])
      self.assertAlmostEqual(allocation.percent_allocation, exp['percent_allocation'], places=2)


if __name__ == '__main__':
  unittest.main()


# To run test, go to top level directory and run:
# coverage run -m unittest tests/test_stock_data_fetcher.py && coverage report -m
