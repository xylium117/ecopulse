@echo off
cd /d "%~dp0"
TITLE EcoPulse Planetary Analytics Engine Launcher
COLOR 0A

echo =======================================================================
echo              EcoPulse Planetary Climate Analytics Engine
echo         Wildfire Damage ^& Flash Flood Inundation AI Platform
echo =======================================================================
echo.

:: Detect Python executable
SET "PYTHON_CMD="
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    SET "PYTHON_CMD=python"
) else (
    py --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        SET "PYTHON_CMD=py"
    ) else (
        echo [ERROR] Python 3.10+ is not found in your system PATH.
        echo Please install Python from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
)

echo [1/4] Checking Python environment and dependencies...
echo Using Python executable: %PYTHON_CMD%
echo Working Directory: %CD%

:: Instant verification if dependencies are already installed
%PYTHON_CMD% -c "import fastapi, uvicorn, numpy, PIL, httpx, mercantile, pydantic" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Python server dependencies already installed and verified.
) else (
    echo Installing missing server packages...
    if exist "server\requirements.txt" (
        %PYTHON_CMD% -m pip install -r server\requirements.txt
    ) else (
        %PYTHON_CMD% -m pip install fastapi "uvicorn[standard]" numpy Pillow pydantic httpx mercantile python-dotenv python-multipart pytest
    )
)

echo.
echo [2/4] Starting FastAPI Telemetry ^& Deep Learning Server on Port 8000...
start "EcoPulse Server API [Port 8000]" cmd /k "cd /d ""%~dp0"" && title EcoPulse Server API && %PYTHON_CMD% -m uvicorn server.app:app --reload --port 8000"

echo [3/4] Starting EcoPulse Planetary Client Dashboard on Port 8080...
start "EcoPulse Client Dashboard [Port 8080]" cmd /k "cd /d ""%~dp0"" && title EcoPulse Client Dashboard && %PYTHON_CMD% -m http.server 8080 --directory client"

echo [4/4] Initializing planetary stream and waiting for servers...
timeout /t 3 /nobreak >nul

echo.
echo Launching EcoPulse in default browser...
start http://localhost:8080

echo.
echo =======================================================================
echo                     EcoPulse is now Active!
echo =======================================================================
echo   * Frontend Web Dashboard: http://localhost:8080
echo   * Backend REST API Docs:  http://localhost:8000/docs
echo   * Health Endpoint:        http://localhost:8000/health
echo =======================================================================
echo Leave this launcher window or the spawned servers open while running.
echo To terminate the application, simply close the server windows.
echo =======================================================================
echo.
pause
