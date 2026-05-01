@echo off
title HS300 System
cd /d "%~dp0"
echo   Starting server at http://127.0.0.1:8501 ...
start "" python -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
timeout /t 10 /nobreak >nul
start http://127.0.0.1:8501
echo   SERVER IS RUNNING!
pause
