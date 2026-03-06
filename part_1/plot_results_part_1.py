# -*- coding: utf-8 -*-
"""
Part 1 — Visualization.

Loads saved results from the results folder and renders amplitude-spectrum
figures with detected carpet regions highlighted.

Expects pairs of files produced by main_part_1.py for each signal:
  results/<stem>_spectrum.csv
  results/<stem>_results.json

Usage
-----
    python plot_results_part_1.py
    python plot_results_part_1.py --input results/ --output figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_RESULTS = Path(__file__).parent / "results"
DEFAULT_FIGURES = Path(__file__).parent / "figures"

FIGSIZE    = (6, 3)
DPI        = 300
BAND_COLOR = "red"
BAND_ALPHA = 0.30

FONTSIZE_TITLE  = 8
FONTSIZE_LABELS = 7
FONTSIZE_TICKS  = 7
FONTSIZE_LEGEND = 7

def plot_from_files(
    spectrum_csv : Path,
    result_json  : Path,
    output_dir   : Path,
) -> Path:
    data        = np.genfromtxt(spectrum_csv, delimiter=",", skip_header=1)
    frequencies = data[:, 0]
    amplitudes  = data[:, 1]

    with open(result_json, encoding="utf-8") as fh:
        meta = json.load(fh)

    source_name = meta["source_file"]
    cpr         = meta["carpet_power_ratio"]
    unit        = meta["unit"]
    regions     = meta["carpet_regions"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(frequencies, amplitudes, color="steelblue", linewidth=0.8, alpha=0.7)

    for i, region in enumerate(regions):
        ax.axvspan(
            region["start_hz"],
            region["end_hz"],
            color=BAND_COLOR,
            alpha=BAND_ALPHA,
            label="Detected carpet" if i == 0 else "_nolegend_",
        )

    ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(f"Amplitude [{unit}]", fontsize=FONTSIZE_LABELS)
    ax.set_title(
        f"Sample {Path(source_name).stem}  |  {len(regions)} carpet(s) detected",
        fontsize=FONTSIZE_TITLE,
    )
    if regions:
        ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = output_dir / f"{Path(source_name).stem}_spectrum.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    log.info("Figure saved → %s", out_path)
    return out_path

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render amplitude-spectrum figures from saved part 1 results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  "-i", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_FIGURES))
    return parser


def main() -> None:
    args        = _build_arg_parser().parse_args()
    results_dir = Path(args.input)
    figures_dir = Path(args.output)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results folder not found: {results_dir}")

    json_files = sorted(results_dir.glob("*_results.json"))
    if not json_files:
        log.warning("No *_results.json files found in '%s'.", results_dir)
        return

    log.info("Found %d result file(s) in '%s'.", len(json_files), results_dir)

    n_ok = n_err = 0
    for json_path in json_files:
        stem     = json_path.name.replace("_results.json", "")
        csv_path = results_dir / f"{stem}_spectrum.csv"
        if not csv_path.exists():
            log.warning("Spectrum CSV not found for '%s', skipping.", json_path.name)
            n_err += 1
            continue
        try:
            plot_from_files(csv_path, json_path, figures_dir)
            n_ok += 1
        except Exception as exc:
            log.error("Failed to plot '%s': %s", json_path.name, exc)
            n_err += 1

    log.info("Done. %d figure(s) saved to '%s'. %d error(s).", n_ok, figures_dir, n_err)


if __name__ == "__main__":
    main()
