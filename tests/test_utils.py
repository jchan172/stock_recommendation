def parameterized(*cases):
  """A decorator to parameterize unittest test methods."""
  def decorator(test_func):
    def wrapper(self):
      for args in cases:
        test_func(self, *args)
    return wrapper
  return decorator
