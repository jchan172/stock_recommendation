import os
import json
from six.moves import urllib
from stock_recommendation.rate_limited_requester import RateLimitedRequester
from stock_recommendation.stock_data_fetcher import *
from stock_recommendation.strategies.basic_example_strategy import basic_strategy

API_KEY = os.environ["TWELVEDATA_API_KEY"]


def send_text_response(event, response_text):
  SLACK_URL = "https://slack.com/api/chat.postMessage"
  channel_id = event["event"]["channel"]
  user = event["event"]["user"]
  bot_token = os.environ["BOT_TOKEN"]
  data = urllib.parse.urlencode({
      "token": bot_token,
      "channel": channel_id,
      "text": response_text,
      "user": user,
      "link_names": True
  })
  data = data.encode("ascii")
  request = urllib.request.Request(SLACK_URL, data=data, method="POST")
  request.add_header("Content-Type", "application/x-www-form-urlencoded")
  res = urllib.request.urlopen(request).read()
  print('res:', res)


def check_message(text):
  text = text.lower()
  requester = RateLimitedRequester(rate_limit=8, per=60)
  stocks = StockDataFetcher(requester, [API_KEY])
  if 'who' in text:
    return 'I am Icarus, a bot that runs algorithms that give stock recommendations.'
  if 'schedule invocation' in text:
    strategy = basic_strategy(stocks)
    return f"""
    FTLT:\n{strategy}\n
    """
  return None


def is_bot(event):
  return 'bot_profile' in event['event']


def lambda_handler(event, context):
  event = json.loads(event["body"])
  print('event after json.loads():', event)
  if not is_bot(event):
    message = check_message(event["event"]["text"])
    if message:
      send_text_response(event, message)

  return {
      'statusCode': 200,
      'body': 'OK'
  }
