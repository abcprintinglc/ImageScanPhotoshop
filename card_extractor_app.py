#!/usr/bin/env python3
"""Simple standalone GUI for business card extraction."""

from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from business_card_extractor import process_document


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Business Card Extractor")
        self.root.geometry("740x360")

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.min_area_var = tk.StringVar(value="0.01")
        self.pdf_dpi_var = tk.StringVar(value="300")
        self.debug_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Choose input and output, then click Run.")

        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Input scan (image/TIFF/PDF):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.input_var, width=76).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse", command=self.pick_input).grid(row=1, column=1)

        ttk.Label(frame, text="Output folder:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(frame, textvariable=self.output_var, width=76).grid(row=3, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(frame, text="Browse", command=self.pick_output).grid(row=3, column=1)

        opts = ttk.Frame(frame)
        opts.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Label(opts, text="Min area ratio:").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.min_area_var, width=12).grid(row=0, column=1, sticky="w", padx=(8, 24))
        ttk.Label(opts, text="PDF DPI:").grid(row=0, column=2, sticky="w")
        ttk.Entry(opts, textvariable=self.pdf_dpi_var, width=12).grid(row=0, column=3, sticky="w", padx=(8, 24))
        ttk.Checkbutton(opts, text="Save debug detection previews", variable=self.debug_var).grid(row=0, column=4, sticky="w")

        self.run_button = ttk.Button(frame, text="Run Extraction", command=self.start)
        self.run_button.grid(row=5, column=0, sticky="w", pady=(16, 0))

        ttk.Label(frame, textvariable=self.status_var, wraplength=700).grid(row=6, column=0, columnspan=2, sticky="w", pady=(16, 0))
        frame.columnconfigure(0, weight=1)

    def pick_input(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Supported", "*.png *.jpg *.jpeg *.tif *.tiff *.pdf *.bmp"), ("All files", "*.*")]
        )
        if path:
            self.input_var.set(path)
            if not self.output_var.get().strip():
                self.output_var.set(str(Path(path).with_suffix("")) + "_cards")

    def pick_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    def start(self) -> None:
        input_raw = self.input_var.get().strip()
        output_raw = self.output_var.get().strip()

        if not input_raw:
            messagebox.showerror("Invalid input", "Please select an input file.")
            return
        if not output_raw:
            messagebox.showerror("Invalid output", "Please select an output folder.")
            return

        input_path = Path(input_raw).expanduser()
        output_path = Path(output_raw).expanduser()

        if not input_path.exists() or not input_path.is_file():
            messagebox.showerror("Invalid input", "Please select a valid input file.")
            return

        try:
            min_area = float(self.min_area_var.get())
            pdf_dpi = int(self.pdf_dpi_var.get())
        except ValueError:
            messagebox.showerror("Invalid options", "Min area ratio must be decimal and PDF DPI must be integer.")
            return

        self.run_button.configure(state="disabled")
        self.status_var.set("Running extraction...")
        thread = threading.Thread(
            target=self._run_job,
            args=(input_path, output_path, min_area, pdf_dpi, self.debug_var.get()),
            daemon=True,
        )
        thread.start()

    def _run_job(self, input_path: Path, output_path: Path, min_area: float, pdf_dpi: int, save_debug: bool) -> None:
        try:
            cards, pages = process_document(
                input_path,
                output_path,
                min_area_ratio=min_area,
                pdf_dpi=pdf_dpi,
                save_debug=save_debug,
            )
            self.root.after(0, self._on_success, cards, pages, output_path, save_debug)
        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    def _on_success(self, cards: int, pages: int, out_path: Path, save_debug: bool) -> None:
        self.run_button.configure(state="normal")
        debug_note = f"\nDebug previews: {out_path / 'debug'}" if save_debug else ""

        if cards == 0:
            msg = (
                f"No cards detected across {pages} page(s).\n\n"
                f"Try Min area ratio = 0.004 and run again.{debug_note}"
            )
            self.status_var.set("Done, but no cards were detected.")
            messagebox.showwarning("No cards detected", msg)
            return

        msg = f"Done. Saved {cards} card(s) from {pages} page(s) to {out_path}.{debug_note}"
        self.status_var.set(msg)
        messagebox.showinfo("Complete", msg)

    def _on_error(self, error: str) -> None:
        self.run_button.configure(state="normal")
        self.status_var.set("Extraction failed.")
        messagebox.showerror("Error", error)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
