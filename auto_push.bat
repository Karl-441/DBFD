@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

REM --- Configuration ---
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "LOGFILE=%SCRIPT_DIR%\auto_git_dbfd.log"
set "TARGET_DIR=%SCRIPT_DIR%"
set "SSH_KEY=d:\Github\secrets\gh_key"
set "REPO_URL=git@github.com:Karl-441/DBFD.git"

REM Configure Git to use the SSH key
set "GIT_SSH_COMMAND=ssh -i "%SSH_KEY%" -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"

REM Navigate to the script directory
cd /d "%TARGET_DIR%"

echo Starting DBFD Auto-Push Service...
echo Logging to: %LOGFILE%
echo.

call :Log "==================================================="
call :Log "Starting Sync Cycle..."

REM --- 1. Validate Git Repository ---
if not exist ".git" (
    call :Log "[ERROR] .git directory not found. Initializing..."
    git init
)

REM --- 2. Configure Remote (Force SSH) ---
REM Check if remote exists and matches SSH format
for /f "tokens=*" %%i in ('git remote get-url origin 2^>nul') do set "CURRENT_REMOTE=%%i"

if "%CURRENT_REMOTE%"=="" (
    call :Log "[INFO] Remote 'origin' not found. Setting to %REPO_URL%..."
    git remote add origin %REPO_URL%
) else (
    if not "%CURRENT_REMOTE%"=="%REPO_URL%" (
        call :Log "[WARN] Current remote is '%CURRENT_REMOTE%'. Switching to SSH '%REPO_URL%'..."
        git remote set-url origin %REPO_URL%
    )
)

REM --- 3. Detect/Create Branch ---
set "BRANCH=main"
git checkout main >nul 2>&1
if %errorlevel% neq 0 (
    call :Log "[INFO] Creating main branch..."
    git checkout -b main >nul 2>&1
)
call :Log "Current Branch: %BRANCH%"

REM --- 4. Add and Commit ---
call :Log "Checking for local changes..."
git add .

REM Check staged changes
git diff --cached --quiet
if %errorlevel% equ 0 (
    call :Log "No changes to commit."
) else (
    call :Log "Changes detected. Files to be committed:"
    
    REM Show status to console and log
    echo ------------------------------------------
    git status --short
    echo ------------------------------------------
    git status --short >> "%LOGFILE%"
    
    call :Log "Committing changes..."
    git commit -m "Auto-commit: %date% %time%" >> "%LOGFILE%" 2>&1
    
    if !errorlevel! equ 0 (
        call :Log "Commit successful."
    ) else (
        call :Log "[ERROR] Commit failed."
        goto :End
    )
)

REM --- 5. Pull (Rebase) ---
call :Log "Pulling updates from remote..."
git pull origin %BRANCH% --rebase >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 (
    call :Log "[WARN] Pull failed (conflicts or no upstream). Attempting push anyway..."
) else (
    call :Log "Pull successful."
)

REM --- 6. Push ---
REM Check what will be pushed
set "HAS_PUSH_CHANGES="
for /f "tokens=*" %%a in ('git diff --stat origin/%BRANCH%..HEAD 2^>nul') do (
    set "HAS_PUSH_CHANGES=1"
)

if defined HAS_PUSH_CHANGES (
    call :Log "Files to be pushed:"
    echo ------------------------------------------
    git diff --stat origin/%BRANCH%..HEAD
    echo ------------------------------------------
    git diff --stat origin/%BRANCH%..HEAD >> "%LOGFILE%"
) else (
    REM Check if ahead
    git rev-list origin/%BRANCH%..HEAD --count > temp_ahead.txt 2>nul
    if exist temp_ahead.txt (
        set /p AHEAD_COUNT=<temp_ahead.txt
        del temp_ahead.txt
    ) else (
        set AHEAD_COUNT=1
    )
    
    if "!AHEAD_COUNT!"=="0" (
        call :Log "Local branch is up to date with remote. Nothing to push."
        goto :End
    )
)

call :Log "Pushing to remote..."
echo [INFO] Using SSH Key: %SSH_KEY%
git push origin %BRANCH%
if %errorlevel% neq 0 (
    call :Log "[ERROR] Push Failed. Check log/network."
) else (
    call :Log "[SUCCESS] Push Successful."
)

:End
call :Log "Process Finished."
call :Log "==================================================="
timeout /t 5 >nul
exit /b

REM --- Subroutines ---
:Log
echo [%date% %time%] %~1
echo [%date% %time%] %~1 >> "%LOGFILE%"
exit /b
