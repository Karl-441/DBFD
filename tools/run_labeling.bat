@echo off
REM DBFD Labeling Tool Launcher

cd /d "%~dp0"

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please create it first.
    pause
    exit /b
)

call venv\Scripts\activate.bat

echo Starting DBFD Labeling Tool...
python tools\labeling_tool.py

pause
