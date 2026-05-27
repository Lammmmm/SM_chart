@echo off
cd /d "%~dp0"
echo Starting Smart Money Dashboard...
python -m streamlit run app.py
pause
