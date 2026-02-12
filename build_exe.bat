@echo off
setlocal

if not exist .venv (
  py -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller

pyinstaller --noconfirm --clean --onefile --windowed --name business-card-extractor card_extractor_app.py

echo.
echo Build complete. EXE is in dist\business-card-extractor.exe
endlocal
