@echo off

REM Smart Chainlit Runner - Automatically finds available port
REM Avoids conflicts with infrastructure services

REM Check if .venv exists and use its Python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

%PYTHON_CMD% chatbot_ui/run_chainlit_ui.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start Chainlit application
    echo Please check the error message above
    pause
    exit /b 1
)


