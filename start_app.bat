@echo off
setlocal

set APP_DIR=d:\Users\kamen.dimitrov\Desktop\SOFTUNI\AI_and_ML_upskill_program\Machine_learning\BG_real_estate_appraisal_helper
set PORT=8891
set URL=http://localhost:%PORT%/

:: Check if already running
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo App is already running on port %PORT%. Opening browser...
    start "" "%URL%"
    exit /b 0
)

:: Check if PostgreSQL is reachable
echo Checking PostgreSQL on port 54891...
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 54891); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PostgreSQL is not running on port 54891.
    echo Please start the postgresql-x64-16 Windows service first.
    pause
    exit /b 1
)

:: Start uvicorn
echo Starting app on %URL% ...
cd /d "%APP_DIR%"
start "Appraisal App" /B python -m uvicorn app.main:app --port %PORT%

:: Wait for app to start (up to 15s)
set /a attempts=0
:wait_loop
timeout /t 1 /nobreak >nul
powershell -Command "try { $r = Invoke-WebRequest -Uri '%URL%' -TimeoutSec 2 -UseBasicParsing; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% == 0 goto app_ready
set /a attempts+=1
if %attempts% LSS 15 goto wait_loop

echo WARNING: App may not have started in time. Opening browser anyway...

:app_ready
start "" "%URL%"
echo App is running at %URL%
