import pytest
import cocoex
from cocoex.exceptions import NoSuchSuiteException


def test_crash_no_dimension():
    """C code crashes when no dimensions remain after filtering.

    See #2181"""
    with pytest.raises(NoSuchSuiteException):
        cocoex.Suite("bbob", "", "dimensions:4")

