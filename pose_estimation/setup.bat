@echo off
echo ========================================================
echo OAK-D Capture Studio Setup
echo ========================================================
echo.

echo [1/3] Creating Python virtual environment...
python -m venv venv

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ========================================================
echo Setup Complete!
echo ========================================================
echo To run the application in the future:
echo 1. Open a terminal in this folder
echo 2. Run: venv\Scripts\activate.bat
echo 3. Run: python capture_gui.py
echo ========================================================
pause
