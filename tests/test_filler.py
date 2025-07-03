"""
This is just a filler test file to test automated github testing.
It can be removed once we write official tests.
"""

import pytest


@pytest.mark.parametrize("variable", [1, "a", 14.5])
def test_filler(variable):
    assert isinstance(str(variable), str)
