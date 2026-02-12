# Business Card Extractor (Standalone Program)

You asked for a **standalone + more stable** workflow (not dependent on Photoshop script prompts). This repo now provides that:

- `card_extractor_app.py` → desktop GUI app (Tkinter).
- `business_card_extractor.py` → command-line engine used by the app.

Both support:

- multiple cards on a page,
- multi-page TIFF and PDF,
- mixed designs and different card sizes (geometry-based detection).

## Quick start (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python card_extractor_app.py
```

In the app:

1. Choose input scan file (`.png/.jpg/.tif/.tiff/.pdf`).
2. Choose output folder.
3. Click **Run Extraction**.
4. Output files are saved as `page_01_card_01.png`, etc.

## Command-line usage

```bash
python business_card_extractor.py input_scan.pdf -o output_cards
```

Useful tuning:

```bash
python business_card_extractor.py scan.jpg -o cards --min-area-ratio 0.006 --pdf-dpi 300
```

## Build a single-file executable (standalone distribution)

If you want an EXE/app you can run without manually launching Python scripts:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed card_extractor_app.py --name business-card-extractor
```

Output binary will be under `dist/`.

## Notes on reliability

- The GUI app directly calls extraction logic; no Photoshop `app.system()` shell bridging needed.
- If detection misses very small cards, lower `--min-area-ratio` (or GUI `Min area ratio`).
- For PDFs, higher DPI improves edge quality but takes longer.

## Optional Photoshop script

`photoshop/RunCardExtraction.jsx` is still included, but the recommended path is the standalone app above.
