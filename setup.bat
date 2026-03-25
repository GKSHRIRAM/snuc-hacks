@echo off
echo ==============================================
echo MarketLens BI Engine - Setup Script
echo ==============================================

echo [1/3] Creating Python Virtual Environment...
python -m venv venv

echo [2/3] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo [3/3] Installing Dependencies...
pip install -r requirements.txt

echo.
echo ==============================================
echo SETUP COMPLETE!
echo ==============================================
echo IMPORTANT: Make sure you have created your .env file with:
echo - FIRECRAWL_API_KEY
echo - GROQ_API_KEY
echo - REDDIT_CLIENT_ID
echo - REDDIT_CLIENT_SECRET
echo.
echo Run "run.bat" to start the server.
pause
