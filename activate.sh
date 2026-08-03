#!/bin/bash
# Hermes Agent environment activation script
# Source this to activate hermes-agent in your shell

export HERMES_HOME="/root/.hermes"
export PATH="/root/.hermes/bin:$HERMES_HOME/.venv/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="/usr/local/share/uv/python"
export UV_PYTHON_BIN_DIR="/usr/local/share/uv/bin"

# Activate the virtual environment
if [ -f "$HERMES_HOME/.venv/bin/activate" ]; then
    source "$HERMES_HOME/.venv/bin/activate"
fi

echo "✅ Hermes Agent activated (v0.19.1)"
echo "   Home: $HERMES_HOME"
echo "   Repo: $HERMES_HOME/hermes-agent"
echo "   Commands: hermes chat, hermes mcp serve, hermes setup, hermes doctor"
