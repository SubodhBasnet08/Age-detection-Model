@echo off
title Age Detection Model App
cd /d "%~dp0"
echo Starting Age Detection Model...
echo.
echo Keep this window open while using the app.
echo If the browser does not open, go to http://127.0.0.1:8000
echo.
start "" cmd /c "timeout /t 4 >nul && start http://127.0.0.1:8000"
"C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -u app.py
echo.
echo The app stopped. Check the message above for the reason.
pause
