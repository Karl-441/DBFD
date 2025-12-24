@echo off
REM DBFD UI Launcher
REM This script activates the virtual environment and starts the GUI

cd /d "%~dp0"

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please create it first.
    pause
    exit /b
)

call venv\Scripts\activate.bat

echo Starting DBFD System...
python ui\gui.py

pause
