@echo off
REM DBFD Labeling Tool Launcher

cd /d "%~dp0"

IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run 'install_dependencies.bat' first to set up the environment.
    pause
    exit /b
)

call venv\Scripts\activate.bat

echo Starting DBFD Labeling Tool...
python tools\labeling_tool.py

pause
