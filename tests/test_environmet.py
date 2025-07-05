# test_environment.py

import pytest
import sys
import os

def test_virtual_environment_active():
    assert hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix), "Test should be run in a virtual environment."

# Optional: check specific environment variable
def test_virtual_environment_variable():
    assert "VIRTUAL_ENV" in os.environ, "VIRTUAL_ENV environment variable is not set."
