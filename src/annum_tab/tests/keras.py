def setup_keras_environment():
    """Minimal setup: uninstall TensorFlow, reinstall Keras standalone, check, print one final message."""

    try:
        import IPython
        ip = get_ipython()
        if ip is not None:
            # Running inside notebook
            ip.run_line_magic("pip", "uninstall -y tensorflow tensorflow-cpu keras")
            ip.run_line_magic("pip", "install keras")
        else:
            raise ImportError
    except (ImportError, NameError):
        # Not in notebook (bash or script)
        import subprocess
        subprocess.run("pip uninstall -y tensorflow tensorflow-cpu keras", shell=True, check=True)
        subprocess.run("pip install keras", shell=True, check=True)

    print("✅ Keras reinstalled, TensorFlow removed if present.")
    print("⚠️ Please restart your Jupyter kernel or shell session if running interactively.")
