# Business Card Extractor (Scan -> Straightened PNGs)

This repository now includes a Python program that:

1. Loads a scanned source document (single image, multipage TIFF, or multipage PDF).
2. Detects rectangular business cards on each page.
3. Corrects perspective (align + straighten).
4. Crops each card.
5. Saves each card as an individual PNG.

## Why this works for mixed layouts

Yes — it can work with:

- multiple cards per page,
- multiple pages,
- different card designs,
- different card sizes.

The detector is geometry-based (contours + quadrilateral checks + non-max suppression), not template-based, so it does not depend on one specific design. You can tune detection with `--min-area-ratio` for very small/very large cards.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python business_card_extractor.py input_scan.pdf -o output_cards
```

Other examples:

```bash
python business_card_extractor.py scan_page.tif -o cards
python business_card_extractor.py scan.jpg -o cards --min-area-ratio 0.006
```

Outputs are named like:

- `page_01_card_01.png`
- `page_01_card_02.png`
- `page_02_card_01.png`

## Photoshop workflow

If you want to trigger extraction from Photoshop, use `photoshop/RunCardExtraction.jsx` (File -> Scripts -> Browse...).

The script asks for:

- path to the Python executable,
- path to `business_card_extractor.py`,
- input file (scan),
- output folder.

It then runs the extractor command and shows completion.

## Notes

- Best results come from high-resolution scans (300 DPI or higher).
- If cards are touching or have weak edges, you may need preprocessing or parameter tuning.
