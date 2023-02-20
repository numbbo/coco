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
    def __init__(self, suite):
        self.suite = suite

    def __str__(self):
        return f"Unknown benchmark suite '{self.suite}'."

    def __repr__(self):
        return f"NoSuchSuiteException('{self.suite}')"


class InvalidProblemException(Exception):
    def __init__(self):
        pass
