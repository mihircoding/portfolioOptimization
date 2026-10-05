@echo off
rem One-click launcher for Windows: installs dependencies if needed,
rem then starts the site and opens it in your browser.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python -m streamlit run streamlit_app.py
