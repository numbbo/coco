import cocoex

def test_crash_no_dimension():
    """C code crashes when no dimensions remain after filtering.

    See #2181"""
    s = cocoex.Suite("bbob", "", "dimensions:4")
    assert len(s) == 0

