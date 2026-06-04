@echo off
cd /d "%~dp0"
echo Starting Smart Money Unified Console...
python -m streamlit run app.py
pause
