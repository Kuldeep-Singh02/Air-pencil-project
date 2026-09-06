@echo off
echo ============================================
echo   Air Pencil - Setup
echo ============================================
echo.

py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.11 nahi mila is PC par.
    echo Pehle Python 3.11 install karo: https://www.python.org/downloads/release/python-3119/
    echo Install karte waqt "Add python.exe to PATH" checkbox tick karna.
    pause
    exit /b
)

echo Python 3.11 mil gaya. Virtual environment ban raha hai...
py -3.11 -m venv venv

echo Environment activate ho raha hai aur libraries install ho rahi hain...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete! Ab "run.bat" pe double-click karo.
echo ============================================
pause
