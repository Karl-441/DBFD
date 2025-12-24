@echo off
REM DBFD UI Launcher
REM This script activates the virtual environment and starts the GUI

cd /d "%~dp0"

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run 'install_dependencies.bat' first to set up the environment.
    pause
    exit /b
)

call venv\Scripts\activate.bat

echo Starting DBFD System...
python ui\gui.py

pause
