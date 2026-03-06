# -*- coding: utf-8 -*-
"""
Part 2 — Visualisation.

Reads JSON and CSV artefacts produced by main_part_2.py and generates
all diagnostic figures and tables.

Figures produced
----------------
  figures/features/violin_features.png      — violin subplots, ordered by Fischer ↓
  figures/features/boxplot_features.png     — box-plot subplots, ordered by Fischer ↓
  figures/features/pearson_heatmap.png      — feature–feature Pearson correlation
  figures/features/spearman_heatmap.png     — feature–feature Spearman correlation
  figures/model/confusion_matrix_cv.png     — summed CV confusion matrix
  figures/test/score_chart.png              — bar chart of test looseness scores
  figures/test/contributions_heatmap.png    — feature-contribution heatmap
  figures/test/<id>_spectrum.png            — amplitude spectrum per test sample

Usage
-----
    python plot_results_part_2.py
    python plot_results_part_2.py --results results/ --output figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_RESULTS = Path(__file__).parent / "results"
DEFAULT_FIGURES = Path(__file__).parent / "figures"

C_HEALTHY = "#2ca02c"
C_LOOSE   = "#d62728"

DPI            = 300
FIGSIZE_WIDE   = (6, 3)
FIGSIZE_SQUARE = (3, 3)
FIGSIZE_GRID   = (4, 3)

FONTSIZE_TITLE         = 7
FONTSIZE_LABELS        = 7
FONTSIZE_TICKS         = 7
FONTSIZE_LEGEND        = 7
FONTSIZE_SUBPLOT_TITLE = 7


def _load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", path)


def _short_name(feat: str) -> str:
    return (feat.replace("horizontal_", "hor_")
                .replace("vertical_",   "ver_")
                .replace("axial_",      "axi_"))


def _features_by_fischer(features: list, fischer: dict) -> list:
    return sorted(features, key=lambda f: fischer.get(f, 0.0), reverse=True)


def _build_per_class_arrays(feature_stats: dict, feat: str) -> tuple:
    h = np.array(feature_stats[feat].get("healthy", []), dtype=float)
    l = np.array(feature_stats[feat].get("structural_looseness", []), dtype=float)
    return h[np.isfinite(h)], l[np.isfinite(l)]


def _heatmap_colorbar_fix(ax: plt.Axes) -> None:
    cbar     = ax.collections[0].colorbar
    ax_pos   = ax.get_position()
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([cbar_pos.x0, ax_pos.y0, cbar_pos.width, ax_pos.height])
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKS)


def plot_violins(feat_data: dict, out_dir: Path) -> None:
    features      = feat_data["selected_features"]
    fischer       = feat_data["fischer"]
    feature_stats = feat_data["feature_stats"]
    ordered       = _features_by_fischer(features, fischer)
    n, ncols      = len(ordered), 3
    nrows         = int(np.ceil(n / ncols))

    fig, axes  = plt.subplots(nrows, ncols, figsize=(FIGSIZE_GRID[0] * ncols, FIGSIZE_GRID[1] * nrows))
    axes_flat  = np.array(axes).flatten() if n > 1 else [axes]

    for ax, feat in zip(axes_flat, ordered):
        h_arr, l_arr = _build_per_class_arrays(feature_stats, feat)
        data_plot, pos_plot, col_plot = [], [], []
        if len(h_arr) >= 4:
            data_plot.append(h_arr); pos_plot.append(0); col_plot.append(C_HEALTHY)
        if len(l_arr) >= 4:
            data_plot.append(l_arr); pos_plot.append(1); col_plot.append(C_LOOSE)
        if data_plot:
            parts = ax.violinplot(data_plot, positions=pos_plot, showmedians=True, showextrema=True, widths=0.7)
            for body, col in zip(parts["bodies"], col_plot):
                body.set_facecolor(col); body.set_alpha(0.55)
            for part in ("cmedians", "cmins", "cmaxes", "cbars"):
                if part in parts:
                    parts[part].set_edgecolor("black"); parts[part].set_linewidth(1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["healthy", "loose"], fontsize=FONTSIZE_TICKS)
        ax.set_title(_short_name(feat), fontsize=FONTSIZE_SUBPLOT_TITLE)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        ax.grid(True, alpha=0.25, axis="y")

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir / "features" / "violin_features.png")


def plot_boxplots(feat_data: dict, out_dir: Path) -> None:
    features      = feat_data["selected_features"]
    fischer       = feat_data["fischer"]
    feature_stats = feat_data["feature_stats"]
    ordered       = _features_by_fischer(features, fischer)
    n, ncols      = len(ordered), 3
    nrows         = int(np.ceil(n / ncols))

    fig, axes  = plt.subplots(nrows, ncols, figsize=(FIGSIZE_GRID[0] * ncols, FIGSIZE_GRID[1] * nrows))
    axes_flat  = np.array(axes).flatten() if n > 1 else [axes]

    for ax, feat in zip(axes_flat, ordered):
        h_arr, l_arr     = _build_per_class_arrays(feature_stats, feat)
        bp_data, bp_labels, bp_colors = [], [], []
        if len(h_arr) >= 2:
            bp_data.append(h_arr); bp_labels.append("healthy"); bp_colors.append(C_HEALTHY)
        if len(l_arr) >= 2:
            bp_data.append(l_arr); bp_labels.append("loose");   bp_colors.append(C_LOOSE)
        if bp_data:
            bp = ax.boxplot(
                bp_data, tick_labels=bp_labels, patch_artist=True,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5),
                widths=0.5,
            )
            for patch, col in zip(bp["boxes"], bp_colors):
                patch.set_facecolor(col); patch.set_alpha(0.55)
        ax.set_title(_short_name(feat), fontsize=FONTSIZE_SUBPLOT_TITLE)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        ax.grid(True, alpha=0.25, axis="y")

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir / "features" / "boxplot_features.png")


def _corr_heatmap(feat_data: dict, method: str, out_dir: Path) -> None:
    features    = feat_data["selected_features"]
    df          = pd.DataFrame(feat_data["samples"])[features].apply(pd.to_numeric, errors="coerce")
    df.dropna(axis=1, how="all", inplace=True)
    feats_avail = df.columns.tolist()
    if len(feats_avail) < 2:
        log.warning("Not enough valid features for %s heatmap.", method)
        return
    corr        = df[feats_avail].corr(method=method)
    short_names = [_short_name(f) for f in feats_avail]
    sz          = max(6, len(feats_avail) * 0.6)
    fig, ax     = plt.subplots(figsize=(sz, sz * 0.88))
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": FONTSIZE_TICKS},
        xticklabels=short_names, yticklabels=short_names,
        linewidths=0.4, square=True,
    )
    _heatmap_colorbar_fix(ax)
    plt.xticks(rotation=45, ha="right", fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)
    plt.tight_layout()
    _save(fig, out_dir / "features" / f"{method}_heatmap.png")


def plot_pearson_heatmap(feat_data: dict, out_dir: Path) -> None:
    _corr_heatmap(feat_data, "pearson", out_dir)


def plot_spearman_heatmap(feat_data: dict, out_dir: Path) -> None:
    _corr_heatmap(feat_data, "spearman", out_dir)


def plot_confusion_matrix_cv(metrics_data: dict, out_dir: Path) -> None:
    fold_results = metrics_data["fold_results"]
    tn = sum(r["TN"] for r in fold_results)
    fp = sum(r["FP"] for r in fold_results)
    fn = sum(r["FN"] for r in fold_results)
    tp = sum(r["TP"] for r in fold_results)
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(
        cm, ax=ax, annot=True, fmt="d", cmap="Blues",
        xticklabels=["healthy", "loose"], yticklabels=["healthy", "loose"],
        linewidths=0.5, cbar=False, annot_kws={"size": FONTSIZE_LABELS},
    )
    ax.set_xlabel("Predicted", fontsize=FONTSIZE_LABELS)
    ax.set_ylabel("Actual",    fontsize=FONTSIZE_LABELS)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    plt.tight_layout()
    _save(fig, out_dir / "model" / "confusion_matrix_cv.png")


def plot_test_scores(results_data: dict, out_dir: Path) -> None:
    results   = results_data["results"]
    threshold = results_data["threshold"]
    scores    = [r["score"] for r in results]
    labels    = [f"{r['sample_id'][:8]}…\n({r['asset_type']})" for r in results]
    colors    = [C_LOOSE if r["is_loose"] else C_HEALTHY for r in results]

    fig, ax = plt.subplots(figsize=(max(FIGSIZE_WIDE[0], len(results) * 1.1), FIGSIZE_WIDE[1]))
    ax.bar(range(len(results)), scores, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3f}")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=FONTSIZE_TICKS, rotation=15, ha="right")
    ax.set_ylabel("P(structural looseness)", fontsize=FONTSIZE_LABELS)
    ax.set_xlabel("Sample", fontsize=FONTSIZE_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Test Set — Structural Looseness Scores\n(red = loose,  green = healthy)",
        fontsize=FONTSIZE_TITLE,
    )
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _save(fig, out_dir / "test" / "score_chart.png")


def plot_contributions_heatmap(results_data: dict, out_dir: Path) -> None:
    results = results_data["results"]
    if not results:
        return

    all_feats  = [c["feature"] for c in results[0]["contributions"]]
    rows, row_labels = [], []
    for r in results:
        contrib_map = {c["feature"]: c["contribution"] for c in r["contributions"]}
        rows.append([contrib_map.get(f, 0.0) for f in all_feats])
        row_labels.append(f"{r['sample_id']} ({'loose' if r['is_loose'] else 'healthy'})")

    mat         = np.array(rows)
    short_feats = [_short_name(f) for f in all_feats]
    fig_h       = max(FIGSIZE_SQUARE[1], len(results) * 0.55)
    fig_w       = max(FIGSIZE_SQUARE[0], len(all_feats) * 0.9)
    fig, ax     = plt.subplots(figsize=(fig_w, fig_h))
    vmax        = float(np.abs(mat).max()) or 1.0
    sns.heatmap(
        mat, ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        annot=True, fmt=".3f", annot_kws={"size": FONTSIZE_TICKS},
        xticklabels=short_feats, yticklabels=row_labels,
        linewidths=0.4,
    )
    _heatmap_colorbar_fix(ax)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Contribution (+ → loose)", fontsize=FONTSIZE_TICKS)
    plt.xticks(rotation=45, ha="right", fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)
    plt.tight_layout()
    _save(fig, out_dir / "test" / "contributions_heatmap.png")
    

ORIENT_STYLES = {
    "horizontal": {"color": "steelblue",  "linestyle": "-"},
    "vertical"  : {"color": "darkorange", "linestyle": "--"},
    "axial"     : {"color": "seagreen",   "linestyle": "-."},
}

CURSOR_COLOR = "grey"
CURSOR_ALPHA = 0.4
CURSOR_LW    = 0.8


def plot_spectra(results_data: dict, spectra_dir: Path, out_dir: Path) -> None:
    """
    Plot amplitude spectra for each test sample up to 11× the rotational frequency,
    with harmonic cursors at 1× – 10×. Reads per-sample spectrum CSVs saved by
    main_part_2.py run_infer.
    """
    ALL_HARMONICS = list(range(1, 11))
    results_map   = {r["sample_id"]: r for r in results_data["results"]}

    for r in results_data["results"]:
        sid       = r["sample_id"]
        rpm       = r["rpm"]
        f_rot     = rpm / 60.0
        f_max     = 11 * f_rot
        csv_path  = spectra_dir / f"{sid}_spectrum.csv"

        if not csv_path.is_file():
            log.warning("Spectrum CSV not found for %s, skipping.", sid)
            continue

        data    = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        freqs   = data[:, 0]
        col_map = {}
        with open(csv_path) as fh:
            header_cols = fh.readline().strip().split(",")
        for i, col in enumerate(header_cols):
            col_map[col.strip()] = i

        label     = r["prediction"].replace("structural_looseness", "loose")
        score     = r["score"]
        is_loose  = r["is_loose"]
        color_tag = C_LOOSE if is_loose else C_HEALTHY

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

        for orient, style in ORIENT_STYLES.items():
            if orient not in col_map:
                continue
            vals = data[:, col_map[orient]]
            if not np.all(np.isnan(vals)):
                mask = freqs <= f_max
                ax.plot(
                    freqs[mask], vals[mask],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=0.8,
                    alpha=0.7,
                    label=orient,
                )

        for h in ALL_HARMONICS:
            ax.axvline(h * f_rot, color=CURSOR_COLOR, linewidth=CURSOR_LW, alpha=CURSOR_ALPHA, linestyle="--")

        ax.set_xlim(0, f_max)
        ax.set_xlabel("Frequency [Hz]", fontsize=FONTSIZE_LABELS)
        ax.set_ylabel("Amplitude [g]",  fontsize=FONTSIZE_LABELS)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
        ax.legend(fontsize=FONTSIZE_LEGEND)
        ax.grid(True, which="both", linestyle="--", alpha=0.2)
        fig.suptitle(
            f"{sid}  |  {r.get('asset_type', '')}  |  {rpm:.0f} RPM"
            f"  |  {label}  (score = {score:.3f})",
            fontsize=FONTSIZE_TICKS,
            color=color_tag,
        )
        fig.tight_layout()

        y_top = ax.get_ylim()[1] * 0.95
        for h in ALL_HARMONICS:
            ax.text(h * f_rot, y_top, f" {h}×", rotation=90, fontsize=FONTSIZE_TICKS,
                    color=CURSOR_COLOR, va="top", ha="center")

        _save(fig, out_dir / "test" / f"{sid}_spectrum.png")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all Part 2 diagnostic figures from saved results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", "-i", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument("--output",  "-o", type=str, default=str(DEFAULT_FIGURES))
    return parser


def main() -> None:
    args        = _build_arg_parser().parse_args()
    results_dir = Path(args.results)
    figures_dir = Path(args.output)

    features_path = results_dir / "features.json"
    if features_path.is_file():
        log.info("Loading %s …", features_path)
        feat_data = _load_json(features_path)
        plot_violins(feat_data, figures_dir)
        plot_boxplots(feat_data, figures_dir)
        plot_pearson_heatmap(feat_data, figures_dir)
        plot_spearman_heatmap(feat_data, figures_dir)
    else:
        log.warning("features.json not found — skipping feature plots.")

    metrics_path = results_dir / "generalization_metrics.json"
    if metrics_path.is_file():
        log.info("Loading %s …", metrics_path)
        metrics_data = _load_json(metrics_path)
        plot_confusion_matrix_cv(metrics_data, figures_dir)
    else:
        log.warning("generalization_metrics.json not found — skipping model plots.")

    results_path = results_dir / "test_results.json"
    if results_path.is_file():
        log.info("Loading %s …", results_path)
        results_data = _load_json(results_path)
        plot_test_scores(results_data, figures_dir)
        plot_contributions_heatmap(results_data, figures_dir)
        plot_spectra(results_data, results_dir / "spectra", figures_dir)
    else:
        log.warning("test_results.json not found — skipping test plots.")

    log.info("All figures saved to: %s", figures_dir)


if __name__ == "__main__":
    main()
