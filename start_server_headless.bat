@echo off
chcp 65001 >nul
title EL Crop Tool Server
set "ROOT=%~dp0"
set "PY=%ROOT%runtime\python.exe"
set "APP=%ROOT%app\app.py"

if not exist "%PY%" (
    echo [ERROR] Python not found. Run 1_download_python.bat first.
    pause & exit /b 1
)
if not exist "%APP%" (
    echo [ERROR] app\app.py not found.
    pause & exit /b 1
)

echo Starting EL Crop Tool server in background...
start /B "" "%PY%" "%APP%" > server.log 2>&1
echo Server started. PID: %ERRORLEVEL%
echo.
echo Access: http://127.0.0.1:15789
echo Logs:   server.log
echo Stop:   taskkill /F /IM python.exe
echo.
