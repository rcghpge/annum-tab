#!/usr/bin/env bash

echo "💡 annum-sdk setup script"

OS="$(uname)"

if [ "$OS" = "FreeBSD" ]; then
    echo "✅ Detected FreeBSD."

    echo "Installing bash..."
    sudo pkg install -y bash

    echo "Installing Python..."
    sudo pkg install -y python

    echo "Installing uv..."
    curl -Ls https://astral.sh/uv/install.sh | sh
    echo "✅ uv installed. Remember to add \$HOME/.local/bin to PATH if not already."

    echo "Creating virtual environment..."
    uv init
    uv venv
    echo "✅ Virtual environment created."

    echo "Activating and installing Python packages..."
    . venv/bin/activate
    pip install --upgrade pip
    uv pip install
    echo "✅ Setup complete! To activate later: source venv/bin/activate"

else
    echo "⚠️ Detected OS: $OS"
    echo "This automated script is only for FreeBSD."
    echo "👉 For macOS:"
    echo "   brew install bash python"
    echo "   curl -Ls https://astral.sh/uv/install.sh | sh"
    echo "👉 For Linux (Debian/Ubuntu):"
    echo "   sudo apt update && sudo apt install -y bash python3 python3-venv"
    echo "   curl -Ls https://astral.sh/uv/install.sh | sh"
    echo "👉 For Windows:"
    echo "   Use Git Bash or WSL. Install Python separately from python.org or via winget."
    echo "👉 For other BSDs (OpenBSD, NetBSD):"
    echo "   pkg_add or pkgin install bash python"
    echo "Then run:"
    echo "   uv venv && source venv/bin/activate"
    echo "   uv pip install -r requirements.txt"
    echo "   or uv pip install"
fi
