"""
test pass on Python package management and environment stack(s) on FreeBSD.
"""

import sys
import os
import platform
import shutil

def detect_python_stack():
    info = {}

    # System and Python basics
    info["system"] = platform.system()
    info["python_executable"] = sys.executable

    # Venv checks
    if hasattr(sys, 'real_prefix'):
        info["venv"] = "virtualenv (real_prefix detected)"
    elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        info["venv"] = "venv (base_prefix detected)"
    elif "VIRTUAL_ENV" in os.environ:
        info["venv"] = f"generic venv (VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']})"
    elif "CONDA_PREFIX" in os.environ:
        info["venv"] = f"conda (CONDA_PREFIX: {os.environ['CONDA_PREFIX']})"
    elif os.environ.get("PDM_ACTIVE"):
        info["venv"] = "PDM detected (PDM_ACTIVE)"
    elif os.environ.get("PIXIE_ACTIVE") or shutil.which("pixi"):
        info["venv"] = "Pixi detected (prefix.dev)"
    elif os.environ.get("POETRY_ACTIVE") or "POETRY_VIRTUALENVS_PATH" in os.environ:
        info["venv"] = "Poetry detected"
    elif os.environ.get("PYENV_VERSION"):
        info["venv"] = f"pyenv detected (PYENV_VERSION: {os.environ['PYENV_VERSION']})"
    else:
        cwd = os.getcwd()
        if any(os.path.exists(os.path.join(cwd, d)) for d in [".venv", "venv", ".pdm-venv"]):
            info["venv"] = "Local venv folder detected (e.g., .venv)"
        else:
            info["venv"] = "No venv detected (system Python?)"

    # Extra package manager binaries
    info["pixi_installed"] = bool(shutil.which("pixi"))
    info["pdm_installed"] = bool(shutil.which("pdm"))
    info["conda_installed"] = bool(shutil.which("conda"))
    info["poetry_installed"] = bool(shutil.which("poetry"))
    info["pipenv_installed"] = bool(shutil.which("pipenv"))
    info["pipx_installed"] = bool(shutil.which("pipx"))
    info["pyenv_installed"] = bool(shutil.which("pyenv"))
    info["uv_installed"] = bool(shutil.which("uv"))

    return info

if __name__ == "__main__":
    info = detect_python_stack()
    print("\n=== Python Environment Stack Detection (FreeBSD-friendly) ===")
    for k, v in info.items():
        print(f"{k}: {v}")
    print("\n✅ Environment detection complete.")
