@echo off
cd /d "%~dp0"

echo Creating venv...
python -m venv venv

echo Activating venv...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Copying models and zones.json from parent directory...
if not exist "models" (
    xcopy /E /I /Y "..\models" "models"
)
if not exist "zones.json" (
    copy "..\zones.json" "zones.json"
)

echo.
echo Setup complete. Run: python main.py
