#!/usr/bin/env python3
"""Extract and deskew business cards from scanned documents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageSequence


@dataclass
class Detection:
    contour: np.ndarray
    score: float
    bbox: Tuple[int, int, int, int]


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(1, int(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(1, int(max(height_a, height_b)))

    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def contour_score(quad: np.ndarray, image_area: float) -> float:
    area = cv2.contourArea(quad)
    if area <= 0:
        return 0.0

    rect = cv2.minAreaRect(quad)
    (w, h) = rect[1]
    if w <= 0 or h <= 0:
        return 0.0

    aspect_ratio = max(w, h) / min(w, h)
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.75) / 1.5)
    area_score = min(1.0, area / (image_area * 0.2))
    rectangularity = max(0.0, min(1.0, area / (w * h)))
    return 0.45 * area_score + 0.35 * aspect_score + 0.2 * rectangularity


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union = aw * ah + bw * bh - inter_area
    return inter_area / union


def non_max_suppression(detections: List[Detection], threshold: float = 0.3) -> List[Detection]:
    detections = sorted(detections, key=lambda x: x.score, reverse=True)
    kept: List[Detection] = []
    for det in detections:
        if all(iou(det.bbox, k.bbox) < threshold for k in kept):
            kept.append(det)
    return kept


def to_quad(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    if box.shape == (4, 2):
        return box.reshape((-1, 1, 2)).astype(np.float32)
    return None


def detect_cards(image: np.ndarray, min_area_ratio: float = 0.01) -> List[np.ndarray]:
    image_area = image.shape[0] * image.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    combined = cv2.bitwise_or(edges, cv2.bitwise_not(adaptive))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * min_area_ratio:
            continue

        quad = to_quad(contour)
        if quad is None:
            continue

        score = contour_score(quad, image_area)
        if score <= 0.15:
            continue

        x, y, w, h = cv2.boundingRect(quad)
        detections.append(Detection(quad, score, (x, y, w, h)))

    kept = non_max_suppression(detections)
    return [d.contour.reshape(4, 2).astype("float32") for d in kept]


def normalize_pdf_bitmap(bitmap: np.ndarray) -> np.ndarray:
    """Normalize pdfium bitmap to BGR without channel swapping artifacts."""
    if bitmap.ndim == 2:
        return cv2.cvtColor(bitmap, cv2.COLOR_GRAY2BGR)

    if bitmap.shape[2] == 4:
        # pdfium commonly returns BGRA by default. Keep BGR order.
        return cv2.cvtColor(bitmap, cv2.COLOR_BGRA2BGR)

    # For 3-channel output, pdfium defaults to BGR byte-order.
    return bitmap.copy()


def load_pages(path: Path, dpi: int = 300) -> List[np.ndarray]:
    ext = path.suffix.lower()
    pages: List[np.ndarray] = []

    if ext == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdfium2. Install dependencies first.") from exc

        pdf = pdfium.PdfDocument(str(path))
        scale = dpi / 72.0
        for i in range(len(pdf)):
            bitmap = pdf[i].render(scale=scale).to_numpy()
            pages.append(normalize_pdf_bitmap(bitmap))
        return pages

    if ext in {".tif", ".tiff"}:
        with Image.open(path) as img:
            for frame in ImageSequence.Iterator(img):
                arr = np.array(frame.convert("RGB"))
                pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return pages

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not open image: {path}")
    return [image]


def save_cards(cards: Sequence[np.ndarray], page: np.ndarray, output_dir: Path, page_index: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    def reading_order_key(pts: np.ndarray) -> Tuple[float, float]:
        return (float(np.mean(pts[:, 1])), float(np.mean(pts[:, 0])))

    count = 0
    for i, pts in enumerate(sorted(cards, key=reading_order_key), start=1):
        warped = four_point_transform(page, pts)
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        out_file = output_dir / f"page_{page_index:02d}_card_{i:02d}.png"
        if cv2.imwrite(str(out_file), warped):
            count += 1
    return count


def save_debug_preview(page: np.ndarray, cards: Sequence[np.ndarray], output_dir: Path, page_index: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = page.copy()
    for idx, pts in enumerate(cards, start=1):
        pts_int = pts.astype(int).reshape((-1, 1, 2))
        cv2.polylines(preview, [pts_int], True, (0, 255, 0), 4)
        center = tuple(np.mean(pts, axis=0).astype(int))
        cv2.putText(preview, str(idx), center, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(str(output_dir / f"page_{page_index:02d}_detections.png"), preview)


def process_document(
    input_path: Path,
    output_dir: Path,
    min_area_ratio: float = 0.01,
    pdf_dpi: int = 300,
    save_debug: bool = False,
) -> Tuple[int, int]:
    pages = load_pages(input_path, dpi=pdf_dpi)
    total_cards = 0

    for page_index, page in enumerate(pages, start=1):
        contours = detect_cards(page, min_area_ratio=min_area_ratio)
        if save_debug:
            save_debug_preview(page, contours, output_dir / "debug", page_index)
        total_cards += save_cards(contours, page, output_dir, page_index)

    return total_cards, len(pages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input scan image, TIFF, or PDF")
    parser.add_argument("-o", "--output", type=Path, default=Path("output_cards"), help="Output directory")
    parser.add_argument("--min-area-ratio", type=float, default=0.01, help="Minimum contour area ratio")
    parser.add_argument("--pdf-dpi", type=int, default=300, help="DPI for PDF rendering")
    parser.add_argument("--debug", action="store_true", help="Write debug detection previews")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total, pages = process_document(args.input, args.output, args.min_area_ratio, args.pdf_dpi, args.debug)
    print(f"Saved {total} card(s) from {pages} page(s) into {args.output}")
    if total == 0:
        print("No cards were detected. Try lower --min-area-ratio (for example 0.004) and use --debug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
