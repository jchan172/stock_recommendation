# A script someone wrote to run a backtest on strategy where the signal was KMLM vs tech.
import datetime
import json
import random
import pandas as pd
import numpy as np
import vectorbt as vbt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def algo(all_data):
  if vbt.RSI.run(all_data['XLK'], window=10, short_name="rsi", ewm=True).rsi[-1] > vbt.RSI.run(all_data['KMLMX'], window=10, short_name="rsi", ewm=True).rsi[-1]:
    return ['TECL']
  else:
    return ['BIL']


kmlmx = None
tecl = None
tecs = None
xlk = None
bil = None

dates = []
k = []
l = []
s = []
x = []
b = []

all_data = pd.DataFrame()

with open('kmlmx.json', 'r') as file:
  kmlmx = json.load(file)

with open('tecl.json', 'r') as file:
  tecl = json.load(file)

with open('tecs.json', 'r') as file:
  tecs = json.load(file)

with open('xlk.json', 'r') as file:
  xlk = json.load(file)

with open('bil.json', 'r') as file:
  bil = json.load(file)

for d, value in zip(kmlmx['charts']['history'][0], kmlmx['charts']['history'][1]):
  ts = datetime.datetime.fromtimestamp(d).date() + datetime.timedelta(days=1)
  k.append(value)
  dates.append(ts)

for value in tecl['charts']['history'][1]:
  l.append(value)

for value in tecs['charts']['history'][1]:
  s.append(value)

for value in xlk['charts']['history'][1]:
  x.append(value)

for value in bil['charts']['history'][1]:
  b.append(value)

all_stonks = [
    'KMLMX', 'TECS', 'TECL', 'XLK', 'BIL'
]

all_unique = list(set(all_stonks))
all_unique.sort()

all_data['KMLMX'] = k
all_data["TECL"] = l
all_data["TECS"] = s
all_data["XLK"] = x
all_data["BIL"] = b
all_data["timestamp"] = dates
all_data.set_index('timestamp', inplace=True)

print(all_data)


# Define the format of the date string
date_format = '%Y-%m-%d'

# Define the date string
sds = '2016-02-01'
# Convert the string to a datetime object
sdo = datetime.datetime.strptime(sds, date_format)
# Extract the date part
start = sdo.date()

# Define the date string
eds = '2020-08-12'
# Convert the string to a datetime object
edo = datetime.datetime.strptime(eds, date_format)
# Extract the date part
end = edo.date()

bt_range = all_data[all_data.index >= start]
bt_range = bt_range[bt_range.index <= end]

print(bt_range)

to_trade = pd.DataFrame(columns=all_unique, dtype="Int64",
                        index=range(len(bt_range.index))).fillna(0)

size = []

last_symbol = list([])
for x in range(len(bt_range.index)):
  if x > 9:
    closes = bt_range.iloc[x]
    picks = algo(bt_range.iloc[:x+1])
    # print(bt_range.index[x], ' - ', picks)
    l = list([])

    for symbol in closes.index:
      if symbol in picks:
        if symbol in last_symbol:
          l.append(0)
        else:
          l.append(1)
      else:
        if symbol in last_symbol:
          l.append(-1)
        else:
          l.append(0)

    last_symbol = picks

    size.append(1/len(picks))
    to_trade.loc[x] = l
  else:
    size.append(0)

s = pd.Series(size)


def strategy(close, to_trade: to_trade):
  signal = np.full(close.shape, np.nan)
  for x in range(len(close)):
    signal[x] = to_trade.loc[x]

  return signal


indicator = vbt.IndicatorFactory(
    class_name="strategy",
    short_name="strategy",
    input_names=["close"],
    param_names=["to_trade"],
    output_names=["signal"]
).from_apply_func(
    strategy,
    to_trade=to_trade
)

results = indicator.run(bt_range)
entries = results.signal == 1.0
exits = results.signal == -1.0

pf = vbt.Portfolio.from_signals(bt_range, entries, exits, freq="d", size=s, size_type='percent', group_by=True,
                                cash_sharing=True, slippage=0.0005, call_seq='auto', allow_partial=True, init_cash=10000)
print(pf.stats())
pf.plot().show()

# Get Quantstats report
import quantstats_lumi as qs
# Extract portfolio daily returns
returns = pf.returns()
# Show the first few rows to ensure correctness
print(returns.head())
# Use quantstats for analysis
qs.extend_pandas()  # Extend pandas functionality
# Example quantstats analysis
print(f"Sharpe Ratio: {qs.stats.sharpe(returns)}")
# Plot performance
qs.reports.full(returns)
