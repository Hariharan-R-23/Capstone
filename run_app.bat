@echo off
title AI Lyric Generation Studio - Capstone Review 2
echo ======================================================================
echo Launching AI Lyric Generation Studio (Capstone Review 2)
echo AR Gladiators - Shiv Nadar University Chennai
echo ======================================================================
echo.

cd /d "%~dp0"

REM Activate Virtual Environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "..\Code\venv\Scripts\activate.bat" (
    call ..\Code\venv\Scripts\activate.bat
)

REM Run Streamlit App
echo Starting Streamlit application on local server...
streamlit run app.py --server.port 8501 --server.headless false

pause
