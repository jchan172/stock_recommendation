# Stock Recommendation

These are scripts that run stock recommendation strategies based on technical indicators. You can run them on your computer using Python, which outputs a message in the terminal indicating the recommendation and an explanation. Verified working on Python 3.12.

To install stuff:
```
pip3 install requests workalendar statistics datetime pytz pytest dotenv numpy vectorbt pandas quantstats_lumi
```

To run strategies, make sure to get a Twelve Data API key and put it in .env, e.g. `API_KEY="ABCD1234"`. Then run:
```
python3 run_strategies.py
```

**Miscellaneous**

To run tests, go to the top level and run this, which will run all tests and generate a coverage report.
```
coverage run -m unittest && coverage report -m
```

If you want to package up this code for something like making a bot that runs strategies and communicates in Slack (deploy in AWS Lambda), run this from top level:
```
./pkg.sh
```

# Backtesting

**Getting historical data**

You need historical data in order to backtest. Get this data using [testfol.io](https://testfol.io/) for all the tickers in the strategy. Here's what you would do:
- Go to [testfol.io](https://testfol.io/) and input date range and add a portfolio and one ticker at 100%
- Before you click "Backtest" button, open up browser developer tools and go to the Network tab.
- Click "Backtest".
- Look for the request to https://testfol.io/api/backtest, then right click and copy the response.
- Open up a text editor on your computer and paste, then save as .json file. Name it the lowercase of the ticker symbol, e.g. SPY data would be `spy.json`.
- Put the .json file into `backtest/stock_data` directory
- Repeat the steps for each ticker that your strategy needs. Make sure they're all the same date ranges.

**Running backtest**

Look at `kmlmx.py` as an example, and create a new file for each strategy you want to backtest. Run it like this:
```
cd backtest
python3 kmlmx.py
```