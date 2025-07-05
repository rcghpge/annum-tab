# Makefile for annum-sdk (FreeBSD only)

.PHONY: install-bash install-uv install-python install-rust

install-bash:
	@echo "⚙️ Installing bash on FreeBSD..."
	sudo pkg install -y bash
	@echo "Check available shell environments available via cat /etc/shells."
	@echo "Check FreeBSD documentation or online for more information."

install-uv:
	@echo "⚙️ Installing uv..."
	curl -Ls https://astral.sh/uv/install.sh | sh
	@echo "✅ uv installed. If needed, add $$HOME/.local/bin to your PATH."

install-python:
	@echo "⚙️ Installing latest Python from pkg..."
	sudo pkg install -y python

install-rust:
        @echo "⚙️ Installing latest Rust from pkg..."
        sudo pkg install -y rust
