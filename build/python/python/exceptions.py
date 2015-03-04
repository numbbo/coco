class NoSuchProblemException(Exception):
    def __init__(self, suit, function_id):
        self.suit = suit
        self.function_id = function_id

    def __str__(self):
        s = "Problem suit '{suit}' lacks a function with id '{id}'."
        return s.format(suit=self.suit, id=self.function_id)

    def __repr__(self):
        s = "NoSuchProblemException('{suit}', {id})"
        return s.format(suit=self.suit, id=self.function_id)

class InvalidProblemException(Exception):
    def __init__(self):
        pass
