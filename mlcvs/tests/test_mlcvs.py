"""
Unit and regression test for the mlcvs package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mlcvs


def test_mlcvs_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mlcvs" in sys.modules
