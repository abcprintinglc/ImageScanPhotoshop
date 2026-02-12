# Business Card Extractor (EXE-first)

Understood: GUI script alone wasn’t enough and you need an actual executable app.

This repo now includes:

- `business_card_extractor.py` (core engine)
- `card_extractor_app.py` (desktop UI)
- `build_exe.bat` (Windows one-click EXE build with PyInstaller)
- `card_extractor_app.spec` (PyInstaller spec)

## What it does

For scans/images/PDF/TIFF:

1. Detect business cards.
2. Perspective-correct / straighten each card.
3. Crop each card.
4. Save each one as PNG (`page_01_card_01.png`, etc.).

Supports multi-page docs, mixed card designs, and different sizes.

## Build the EXE (Windows)

Open Command Prompt in the repo and run:

```bat
build_exe.bat
```

EXE output:

- `dist\business-card-extractor.exe`

## Run the EXE

1. Launch `business-card-extractor.exe`.
2. Choose input file.
3. Choose output folder.
4. Click **Run Extraction**.

If no cards are found, app now warns clearly and suggests tuning.

## Better diagnostics (important)

The app now defaults to **Save debug detection previews**.

- Debug images are saved to `output_folder\debug\page_XX_detections.png`.
- These show green outlines where detections happened.
- If output is empty, open debug images to see whether detection failed or cards were too small.

## Tuning when no output appears

- Lower **Min area ratio** (try `0.004`).
- Increase **PDF DPI** (try `400`).
- Ensure scans are high contrast and cards are separated.

## CLI usage (optional)

```bash
python business_card_extractor.py input_scan.pdf -o output_cards --debug
```

If it prints “No cards were detected”, try:

```bash
python business_card_extractor.py input_scan.pdf -o output_cards --min-area-ratio 0.004 --pdf-dpi 400 --debug
```

## Photoshop script

Photoshop bridge remains optional (`photoshop/RunCardExtraction.jsx`), but EXE is the recommended workflow.


## Photoshop plugin workflow (requested)

Use `photoshop/CardAlignCropExport.jsx` for a Photoshop-native workflow:

1. Open image/card in Photoshop.
2. Make a selection around the card/image area.
3. Run `File -> Scripts -> Browse...` and choose `CardAlignCropExport.jsx`.
4. Enter rotation angle (for straightening), choose PNG-8 color count (smaller file), choose output folder.
5. Script crops to your selection with **delete=false** (hidden pixels preserved), then exports optimized PNG-8.

Notes:
- This avoids the blue color-shift issue from the previous external pipeline by exporting directly from Photoshop.
- For smallest files, try 32 or 64 colors.
