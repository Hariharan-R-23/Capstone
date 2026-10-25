@echo off
title AI Lyric Generation Studio - Capstone Review 2
echo ======================================================================
echo Launching AI Lyric Generation Studio (Capstone Review 2)
echo AR Gladiators - Shiv Nadar University Chennai
echo ======================================================================
echo.

cd /d "%~dp0"

REM Explicit Python in venv running Streamlit module
if exist "%~dp0venv\Scripts\python.exe" (
    echo Starting Streamlit application on local server using virtual environment...
    "%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.port 8501 --server.headless false
) else if exist "%~dp0..\Code\venv\Scripts\python.exe" (
    echo Starting Streamlit application on local server using virtual environment...
    "%~dp0..\Code\venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.port 8501 --server.headless false
) else (
    echo [ERROR] Virtual environment python.exe not found!
    echo Please make sure the venv directory exists.
)

pause
