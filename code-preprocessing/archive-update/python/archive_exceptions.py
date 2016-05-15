class PreprocessingException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class PreprocessingWarning(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str("PRE-PROCESSING WARNING: " + self.value)        