from sys import float_info

def about_equal(a, b):
    """
    Return True if the floating point number ${a} and ${b} are about equal.
    """
    absolute_error = abs(a - b)
    larger = a if abs(a) > abs(b) else b
    relative_error = abs((a - b) / larger);

    if absolute_error < (2 * float_info.min):
        return True;
    return relative_error < 1e-6;
