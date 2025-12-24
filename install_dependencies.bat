@echo off
cd /d "%~dp0"

echo ===================================================
echo   DBFD Dependency Installer
echo ===================================================

REM 1. Check for Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Please ensure Python is installed and in your PATH.
        pause
        exit /b
    )
) else (
    echo [INFO] Virtual environment 'venv' already exists.
)

REM 2. Activate Virtual Environment
call venv\Scripts\activate.bat

REM 3. Install Dependencies
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

if exist "requirements.txt" (
    echo [INFO] Installing requirements from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b
    )
    echo [SUCCESS] Dependencies installed successfully.
) else (
    echo [WARNING] requirements.txt not found!
)

echo.
echo ===================================================
echo   Setup Complete!
echo   You can now run 'run_ui.bat' to start the application.
echo ===================================================
pause
