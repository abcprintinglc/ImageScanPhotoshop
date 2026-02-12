@echo off
setlocal

where py >nul 2>nul
if %errorlevel% neq 0 (
  echo ERROR: Python launcher ^(py^) not found.
  exit /b 1
)

if not exist .venv (
  py -m venv .venv
)

call .venv\Scripts\activate
if %errorlevel% neq 0 (
  echo ERROR: Could not activate virtual environment.
  exit /b 1
)

python -m pip install --upgrade pip
if %errorlevel% neq 0 exit /b 1

pip install -r requirements.txt pyinstaller
if %errorlevel% neq 0 exit /b 1

python -m PyInstaller --noconfirm --clean card_extractor_app.spec
if %errorlevel% neq 0 exit /b 1

echo.
echo Build complete: dist\business-card-extractor.exe
endlocal
