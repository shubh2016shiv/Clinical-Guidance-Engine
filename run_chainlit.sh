#!/bin/bash
# Smart Chainlit Runner - Automatically finds available port
# Avoids conflicts with infrastructure services

# Check if .venv exists and use its Python
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD chatbot_ui/run_chainlit_ui.py "$@"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to start Chainlit application"
    echo "Please check the error message above"
    exit 1
fi


