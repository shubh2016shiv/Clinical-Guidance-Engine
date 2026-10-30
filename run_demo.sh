#!/bin/bash
# Starts the Healthcare Assistant web interface (NiceGUI)

# Check if .venv exists and use its Python
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD healthcare_assistant_web/main.py "$@"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to start the web application"
    echo "Please check the error message above"
    exit 1
fi


