@echo off
cd /d "%~dp0"
set PYTHONPATH=src
"C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
pause