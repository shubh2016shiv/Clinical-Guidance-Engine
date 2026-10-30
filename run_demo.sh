#!/bin/bash
# Starts the Healthcare Assistant web interface (NiceGUI)

# Check if .venv exists and use its Python
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python"
fi

# Make sure Milvus and Redis are up before the app starts. This does nothing if
# they are already healthy; otherwise it starts them (the first run pulls images).
$PYTHON_CMD infrastructure/manage_infrastructure.py --ensure

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Infrastructure is not available, so the web application was not started"
    echo "Make sure Docker is running, then try again"
    exit 1
fi

$PYTHON_CMD healthcare_assistant_web/main.py "$@"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Failed to start the web application"
    echo "Please check the error message above"
    exit 1
fi


