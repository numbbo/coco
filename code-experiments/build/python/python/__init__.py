import utilities
try:
    from ._interface import Problem, Benchmark, benchmarks
except Exception as e:
    print("numbbo/code-experiments/build/python/python/__init__.py: could not import '_interface'", e)
    from .interface import Problem, Benchmark, benchmarks
  
# from .utilities import about_equal
# from .exceptions import NoSuchProblemException, InvalidProblemException
