@echo off

REM Starts the Healthcare Assistant web interface (NiceGUI)

REM Check if .venv exists and use its Python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

%PYTHON_CMD% healthcare_assistant_web/main.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the web application
    echo Please check the error message above
    pause
    exit /b 1
)


