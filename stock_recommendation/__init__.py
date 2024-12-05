import importlib
import pkgutil
import inspect

# Automatically import all functions from submodules
for module_info in pkgutil.iter_modules(__path__):
  # Dynamically import the module
  module = importlib.import_module(f"{__name__}.{module_info.name}")

  # Get all functions and variables from the module and add them to the current namespace
  for name, obj in inspect.getmembers(module):
    if inspect.isfunction(obj) or inspect.isclass(obj):
      globals()[name] = obj
