class NoSuchProblemException(Exception):
    def __init__(self, suite, function_id):
        self.suite = suite
        self.function_id = function_id

    def __str__(self):
        s = "Problem suite '{suite}' lacks a function with id '{id}'."
        return s.format(suite=self.suite, id=self.function_id)

    def __repr__(self):
        s = "NoSuchProblemException('{suite}', {id})"
        return s.format(suite=self.suite, id=self.function_id)
    
class NoSuchSuiteException(Exception):
    pass

class InvalidProblemException(Exception):
    def __init__(self):
        pass
