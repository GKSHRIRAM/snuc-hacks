@echo off
echo ==============================================
echo Starting MarketLens BI Engine...
echo ==============================================

if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python main.py
pause
