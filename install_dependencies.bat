@echo off
REM DBFD Dependency Installer
REM This script creates a virtual environment and installs all required dependencies

cd /d "%~dp0"

echo [INFO] Checking for virtual environment...
IF NOT EXIST "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment. Please ensure python is installed and in your PATH.
        pause
        exit /b
    )
) ELSE (
    echo [INFO] Virtual environment found.
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing dependencies from requirements.txt...
IF EXIST "requirements.txt" (
    pip install -r requirements.txt
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b
    )
) ELSE (
    echo [WARNING] requirements.txt not found! Skipping dependency installation.
)

echo [SUCCESS] Installation complete!
echo You can now run the application using 'run_ui.bat'.
pause
