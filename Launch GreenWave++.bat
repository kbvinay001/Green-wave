@echo off
title Green Wave++ Launcher
color 0A

echo.
echo  ====================================================
echo   GREEN WAVE++ -- Emergency Vehicle Preemption
echo  ====================================================
echo.

:: Set working directory to the greenwave folder
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Check if Node.js is installed (for React dashboard)
node --version >nul 2>&1
if errorlevel 1 (
    echo [WARN]  Node.js not found - dashboard will not start
    echo         Install Node.js 18+ from https://nodejs.org/
    set NO_UI=1
) else (
    set NO_UI=0
)

:: Install npm packages if needed
if %NO_UI%==0 (
    if not exist "ui\node_modules\" (
        echo [INFO]  Installing React dashboard dependencies...
        echo         (This only happens once)
        echo.
        cd ui
        call npm install --silent
        cd ..
        echo [OK]    npm install complete
        echo.
    )
)

:: Ask demo or live mode
echo  Choose mode:
echo    1 = DEMO MODE  (synthetic data - no hardware needed)
echo    2 = LIVE MODE  (real mic + camera)
echo.
set /p CHOICE="  Enter 1 or 2 [default: 1]: "
if "%CHOICE%"=="" set CHOICE=1
if "%CHOICE%"=="2" (
    set DEMO_FLAG=
    echo [INFO]  Starting in LIVE mode
) else (
    set DEMO_FLAG=--demo
    echo [INFO]  Starting in DEMO mode
)

echo.
echo  Starting backend server...
start "GreenWave++ Backend" cmd /k "cd /d "%~dp0" && python run.py %DEMO_FLAG% && pause"

:: Give the backend a moment to start before opening the browser
timeout /t 3 /nobreak >nul

:: Start Vite dashboard if Node is available
if %NO_UI%==0 (
    echo  Starting React dashboard...
    start "GreenWave++ Dashboard" cmd /k "cd /d "%~dp0\ui" && npm run dev && pause"

    :: Give Vite time to compile
    timeout /t 4 /nobreak >nul

    :: Open browser
    echo  Opening dashboard in browser...
    start "" "http://localhost:5173"
) else (
    echo  Dashboard skipped (Node.js not installed)
    echo  Backend API: http://localhost:8000
    echo  WebSocket:   ws://localhost:8000/ws
)

echo.
echo  ====================================================
echo   Green Wave++ is running!
echo.
echo   Dashboard : http://localhost:5173
echo   Backend   : http://localhost:8000
echo   Status    : http://localhost:8000/status
echo.
echo   Close this window to keep the server running.
echo   Close the Backend/Dashboard windows to stop them.
echo  ====================================================
echo.
pause
