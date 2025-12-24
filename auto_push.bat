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
git branch --list %BRANCH% >nul 2>&1
if %errorlevel% neq 0 (
    call :Log "[INFO] Creating local branch %BRANCH%..."
    git checkout -b %BRANCH% >nul 2>&1
) else (
    git checkout %BRANCH% >nul 2>&1
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

REM --- 5. Fetch and Pull ---
call :Log "Fetching updates from remote..."
git fetch origin >> "%LOGFILE%" 2>&1

REM Check if remote branch exists
git rev-parse --verify origin/%BRANCH% >nul 2>&1
if %errorlevel% equ 0 (
    call :Log "Pulling updates from remote..."
    git pull origin %BRANCH% --rebase >> "%LOGFILE%" 2>&1
    if !errorlevel! neq 0 (
        call :Log "[WARN] Pull failed (conflicts?). Attempting push anyway..."
    ) else (
        call :Log "Pull successful."
    )
) else (
    call :Log "[INFO] Remote branch origin/%BRANCH% does not exist yet. Skipping pull."
)

REM --- 6. Push ---
set "SHOULD_PUSH=0"

REM Check if remote exists to diff against
git rev-parse --verify origin/%BRANCH% >nul 2>&1
if %errorlevel% equ 0 (
    REM Check for unpushed commits
    git diff --stat origin/%BRANCH%..HEAD > temp_diff.txt 2>nul
    for %%A in (temp_diff.txt) do if %%~zA gtr 0 set "SHOULD_PUSH=1"
    
    if "!SHOULD_PUSH!"=="1" (
        call :Log "Files to be pushed:"
        type temp_diff.txt
        type temp_diff.txt >> "%LOGFILE%"
    ) else (
        call :Log "Local branch is up to date with remote. Nothing to push."
    )
    del temp_diff.txt
) else (
    REM Remote doesn't exist, must push
    set "SHOULD_PUSH=1"
    call :Log "[INFO] First push to new remote branch."
)

if "!SHOULD_PUSH!"=="1" (
    call :Log "Pushing to remote..."
    echo [INFO] Using SSH Key: %SSH_KEY%
    
    REM Use -u to set upstream tracking
    git push -u origin %BRANCH% >> "%LOGFILE%" 2>&1
    
    if !errorlevel! neq 0 (
        call :Log "[ERROR] Push Failed. Check log/network."
        echo [ERROR] Push Failed. Check auto_git_dbfd.log for details.
    ) else (
        call :Log "[SUCCESS] Push Successful."
    )
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
