# test_dependencies.py

import pytest

def test_requests_import():
    try:
        import requests
    except ImportError:
        pytest.fail("The 'requests' package is not installed.")

def test_requests_get():
    try:
        import requests
        response = requests.get("https://httpbin.org/get")
        assert response.status_code == 200, "requests.get failed or returned an unexpected status code"
    except ImportError:
        pytest.fail("The 'requests' package is not installed.")
    except Exception as e:
        pytest.fail(f"requests.get failed with an error: {e}")

