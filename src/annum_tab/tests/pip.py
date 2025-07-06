# Tensorflow/Keras pipeline tests/debugging
import subprocess

subprocess.run("pip uninstall -y tensorflow tensorflow-cpu keras", shell=True, check=True)
subprocess.run("pip install keras", shell=True, check=True)
subprocess.run("ls .venv/lib/python3.13/site-packages/ | grep tensorflow || true", shell=True, check=True)

