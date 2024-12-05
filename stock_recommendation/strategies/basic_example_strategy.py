from stock_recommendation.stock_data_fetcher import (
    Allocation,
    StockDataFetcher,
    StockRecommendation
)


def basic_strategy(stocks: StockDataFetcher) -> StockRecommendation:
  qqq_rsi = stocks.rsi("QQQ", 10)
  if qqq_rsi < 30:
    return StockRecommendation([Allocation("TQQQ", 100)], f"QQQ 10d RSI {qqq_rsi} < 30, which is oversold. Buy TQQQ (3x QQQ).")

  return StockRecommendation([Allocation("QQQ", 100)], f"QQQ 10d RSI {qqq_rsi} >= 30, so normal conditions. Buy QQQ.")