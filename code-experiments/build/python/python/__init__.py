import utilities
try:
    from ._interface import Suite, Observer
except Exception as e:
    print("numbbo/code-experiments/build/python/python/__init__.py: could not import '_interface', trying 'interface'", e)
    from .interface import Suite, Observer
  
# from .utilities import about_equal
# from .exceptions import NoSuchProblemException, InvalidProblemException
