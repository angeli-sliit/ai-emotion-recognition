@echo off
REM Quick Start Script for AI Emotion Recognition Web App (Windows)

echo 🚀 Starting AI Emotion Recognition Web Application...
echo.

REM Check if model file exists in parent directory
if not exist "..\mod_my_model01.keras" (
    echo ⚠️  Warning: Model file '..\mod_my_model01.keras' not found!
    echo    Please train the model first using face022.ipynb in the parent directory
    echo.
    pause
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -q -r requirements.txt

REM Run Streamlit app
echo.
echo ✅ Starting Streamlit application...
echo 🌐 Open your browser at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run webapp.py

pause

