import time
from collections import deque
import requests


class RateLimitedRequester:
  """Makes API requests in a rate limited manner to work under an API's rate limits."""
  def __init__(self, rate_limit=5, per=60):
    """  
    Initializes the RateLimitedRequester.  

    :param rate_limit: Maximum number of requests allowed per time period.  
    :param per: Time period in seconds for the rate limit.  
    """
    self.rate_limit = rate_limit
    self.per = per
    self.request_times = deque()

  def _is_rate_limited(self):
    """Check if the request is within the rate limit."""
    current_time = time.time()

    # Remove timestamps that are out of the time window
    while self.request_times and self.request_times[0] <= current_time - self.per:
      self.request_times.popleft()

    # If the length of the deque is equal to the rate limit, we are rate limited
    return len(self.request_times) >= self.rate_limit

  def get(self, url, **kwargs):
    """  
    Performs a GET request.  

    :param url: The URL to send the GET request to.  
    :param kwargs: Additional arguments to pass to requests.get().  
    :return: The response of the GET request.  
    """
    # Wait until we can make a request
    while self._is_rate_limited():
      time.sleep(1)  # Sleep for a short time and check again

    # Make the request
    response = requests.get(url, timeout=10, **kwargs)

    # Record the request time
    self.request_times.append(time.time())

    return response
