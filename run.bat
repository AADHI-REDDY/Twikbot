@echo off
REM Twitter Fake Account Detection - Run Script
REM This script starts the Streamlit application

echo ========================================
echo Twitter Fake Account Detection
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Starting Streamlit application...
echo.
echo The dashboard will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================
echo.

REM Start Streamlit
streamlit run app.py

pause
