@echo off

REM Starts the Healthcare Assistant web interface (NiceGUI)

REM Check if .venv exists and use its Python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    set PYTHON_CMD=python
)

REM Make sure Milvus and Redis are up before the app starts. This does nothing if
REM they are already healthy; otherwise it starts them (the first run pulls images).
%PYTHON_CMD% infrastructure\manage_infrastructure.py --ensure

if errorlevel 1 (
    echo.
    echo ERROR: Infrastructure is not available, so the web application was not started
    echo Make sure Docker is running, then try again
    pause
    exit /b 1
)

%PYTHON_CMD% healthcare_assistant_web/main.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the web application
    echo Please check the error message above
    pause
    exit /b 1
)


