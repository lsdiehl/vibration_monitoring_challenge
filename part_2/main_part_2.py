# -*- coding: utf-8 -*-
"""
Part 2 — Structural looseness detection.

End-to-end pipeline: feature extraction → generalization estimation →
final model training → inference.

Outputs
-------
  results/features.json               feature matrix, AUROC, Spearman, Fischer
  results/generalization_metrics.json nested CV performance estimates
  results/model.joblib                trained model bundle
  results/training_info.json          best hyperparameters and decision threshold
  results/test_results.json           predictions, scores, contributions
  results/spectra/<id>_spectrum.csv   per-sample amplitude spectra (3 orientations)

Usage
-----
    python main_part_2.py
    python main_part_2.py --config config_part_2.json
    python main_part_2.py --run extract
    python main_part_2.py --run generalize
    python main_part_2.py --run train
    python main_part_2.py --run infer
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from scipy.signal import welch
from scipy.stats import loguniform, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CONFIG     = Path(__file__).parent / "config_part_2.json"
DEFAULT_TRAIN_DIR  = Path(__file__).parent / "data" / "train"
DEFAULT_TEST_DIR   = Path(__file__).parent / "data" / "test"
DEFAULT_TRAIN_META = Path(__file__).parent / "data" / "train_metadata.csv"
DEFAULT_TEST_META  = Path(__file__).parent / "data" / "test_metadata.csv"
DEFAULT_OUTPUT     = Path(__file__).parent / "results"

@dataclass
class Window:
    name: Literal["hanning"] = "hanning"
    size: int = 0
    _array: np.ndarray = field(default_factory=lambda: np.array([]), repr=False, init=False)

    def __post_init__(self) -> None:
        if self.size > 0:
            self._build()

    def _build(self) -> None:
        if self.name == "hanning":
            self._array = np.hanning(self.size)
        else:
            raise ValueError(f"Unsupported window: '{self.name}'.")

    def resize(self, size: int) -> None:
        self.size = size
        self._build()

    @property
    def array(self) -> np.ndarray:
        if self._array.size == 0:
            raise RuntimeError("Window has not been built. Set size > 0.")
        return self._array

    @property
    def amplitude_correction_factor(self) -> float:
        return float(self.size / self.array.sum())


class Spectrum(ABC):
    def __init__(
        self,
        frequencies: np.ndarray,
        values: np.ndarray,
        window: Window,
        unit: str = "",
    ) -> None:
        self.frequencies = frequencies
        self.values      = values
        self.window      = window
        self.unit        = unit

    @property
    def freq_resolution(self) -> float:
        if len(self.frequencies) < 2:
            return float("nan")
        return float(self.frequencies[1] - self.frequencies[0])

    @property
    def nyquist_hz(self) -> float:
        return float(self.frequencies[-1])

    @abstractmethod
    def unit_label(self) -> str: ...


class AmplitudeSpectrum(Spectrum):
    def unit_label(self) -> str:
        return f"Amplitude [{self.unit}]"

    @classmethod
    def from_wave(cls, wave: "Wave", window: Window) -> "AmplitudeSpectrum":
        sig = np.asarray(wave.signal, dtype=float)
        N   = len(sig)
        window.resize(N)
        acf = window.amplitude_correction_factor
        X   = np.fft.rfft(sig * window.array)
        f   = np.fft.rfftfreq(N, d=wave.dt)
        amp = np.abs(X) / N * acf
        amp[1:-1] *= 2
        return cls(frequencies=f, values=amp, window=window, unit=wave.unit)


class Wave(BaseModel):
    time     : List[float] = Field(..., description="Time points of the wave")
    signal   : List[float] = Field(..., description="Signal values")
    quantity : str         = Field("acceleration", description="Physical quantity of the signal")
    unit     : str         = Field("g",            description="Measurement unit")

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _check_lengths_and_size(self) -> "Wave":
        if len(self.time) != len(self.signal):
            raise ValueError(
                f"'time' and 'signal' must have the same length "
                f"({len(self.time)} vs {len(self.signal)})."
            )
        if len(self.time) < 2:
            raise ValueError("Wave must contain at least 2 samples.")
        return self

    @computed_field
    @property
    def dt(self) -> float:
        return float(np.diff(self.time).mean())

    @computed_field
    @property
    def duration(self) -> float:
        t = np.asarray(self.time)
        return float(t[-1] - t[0])

    @computed_field
    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt


class Config(BaseModel):
    wave_quantity   : Literal["acceleration", "velocity", "displacement"] = Field(
        "acceleration", description="Default physical quantity for loaded waves"
    )
    wave_unit       : str          = Field("g",      description="Default measurement unit")
    band_half_width : float        = Field(0.1,      description="Half-width around each harmonic [fraction of f_rot]")
    harmonics       : List[int]    = Field(default_factory=lambda: list(range(3, 11)),
                                          description="Harmonic orders used for feature extraction")
    orientations    : List[str]    = Field(default_factory=lambda: ["horizontal", "vertical", "axial"],
                                          description="Vibration orientations")
    n_outer         : int          = Field(5,        description="Outer CV folds")
    n_inner         : int          = Field(3,        description="Inner CV folds")
    n_iter          : int          = Field(100,      description="Random hyperparameter draws")
    f_beta          : int          = Field(1,        description="Beta for F-beta threshold tuning")
    random_state    : int          = Field(42,       description="Random seed")
    window_name     : Literal["hanning"] = Field("hanning", description="FFT window type")

    @field_validator("n_outer", "n_inner", "n_iter")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be ≥ 1, got {v}.")
        return v

    @field_validator("band_half_width")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"'band_half_width' must be > 0, got {v}.")
        return v

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        log.info("Config loaded from '%s'.", path.name)
        return cls(**data)

    @property
    def per_orient_features(self) -> List[str]:
        return ["harmonic_flatness", "harmonic_irregularity", "harmonic_coeff_variation"]

    @property
    def feature_list(self) -> List[str]:
        return [f"{o}_{f}" for o in self.orientations for f in self.per_orient_features]



def _band_max(amp: np.ndarray, f: np.ndarray, fc: float, bw: float) -> float:
    mask = (f >= fc - bw) & (f <= fc + bw)
    return float(amp[mask].max()) if mask.any() else 0.0


def _parse_orientation(s) -> dict:
    return s if isinstance(s, dict) else ast.literal_eval(s)


def extract_features_from_wave(wave: Wave, rpm: float, orient: str, cfg: Config) -> dict:
    f_rot = rpm / 60.0
    p     = f"{orient}_"
    nan_row = {f"{p}{f}": np.nan for f in cfg.per_orient_features}

    if f_rot <= 0 or len(wave.signal) < 4:
        return nan_row

    sig    = np.asarray(wave.signal, dtype=float)
    bw     = cfg.band_half_width * f_rot
    window = Window(name=cfg.window_name)
    window.resize(len(sig))
    acf    = window.amplitude_correction_factor
    X      = np.fft.rfft(sig * window.array)
    f_arr  = np.fft.rfftfreq(len(sig), d=wave.dt)
    amp    = np.abs(X) / len(sig) * acf
    amp[1:-1] *= 2

    h_amps = {h: _band_max(amp, f_arr, h * f_rot, bw) for h in cfg.harmonics}
    vals   = np.maximum(np.array([h_amps[h] for h in cfg.harmonics]), 0.0)

    feats = {}
    valid = vals[vals > 0]
    if len(valid) >= 2:
        geo = float(np.exp(np.mean(np.log(valid))))
        am  = float(np.mean(valid))
        feats[f"{p}harmonic_flatness"] = geo / am if am > 0 else np.nan
    else:
        feats[f"{p}harmonic_flatness"] = np.nan

    sq = float(np.sum(vals ** 2))
    if sq > 0 and len(vals) >= 2:
        feats[f"{p}harmonic_irregularity"] = float(np.sum((vals[:-1] - vals[1:]) ** 2) / sq)
    else:
        feats[f"{p}harmonic_irregularity"] = np.nan

    feats[f"{p}harmonic_coeff_variation"] = (
        float(vals.std() / vals.mean()) if vals.mean() > 0 else np.nan
    )
    return feats


def extract_features_from_trio(
    wave_hor: Optional[Wave],
    wave_ver: Optional[Wave],
    wave_axi: Optional[Wave],
    rpm: float,
    cfg: Config,
) -> dict:
    orient_feats = {}
    for orient, wave in [("horizontal", wave_hor), ("vertical", wave_ver), ("axial", wave_axi)]:
        if wave is not None:
            orient_feats[orient] = extract_features_from_wave(wave, rpm, orient, cfg)

    row = {}
    for fd in orient_feats.values():
        row.update(fd)
    return {k: row.get(k, np.nan) for k in cfg.feature_list}


def _load_train_waves(csv_path: Path, orient_meta: dict, cfg: Config) -> dict:
    df         = pd.read_csv(csv_path)
    axis_to_ch = {"axisX": "Ch1 Y-Axis", "axisY": "Ch2 Y-Axis", "axisZ": "Ch3 Y-Axis"}
    ch_map     = {axis_to_ch[k]: v for k, v in orient_meta.items() if k in axis_to_ch}
    orient_map = {v: k for k, v in ch_map.items()}
    time_arr   = df["X-Axis"].values.tolist()
    waves      = {}
    for orient in cfg.orientations:
        col = orient_map.get(orient)
        if col is None or col not in df.columns:
            waves[orient] = None
        else:
            waves[orient] = Wave(time=time_arr, signal=df[col].values.tolist())
    return waves


def _load_test_waves(csv_path: Path, orient_meta: dict, cfg: Config) -> dict:
    df             = pd.read_csv(csv_path)
    axis_to_col    = {"axisX": "x", "axisY": "y", "axisZ": "z"}
    waves          = {}
    for orient in cfg.orientations:
        col = next(
            (axis_to_col[k]
             for k, v in orient_meta.items()
             if v == orient and axis_to_col.get(k) in df.columns),
            None,
        )
        if col is None:
            waves[orient] = None
        else:
            waves[orient] = Wave(time=df["t"].values.tolist(), signal=df[col].values.tolist())
    return waves


def _make_pipe(random_state: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(solver="saga", max_iter=10000, random_state=random_state)),
    ])


def _param_dist() -> dict:
    return {
        "lr__C":       loguniform(1e-3, 1e3),
        "lr__penalty": ["l1", "l2"],
    }


def _best_threshold(y_true: np.ndarray, proba: np.ndarray, beta: int) -> float:
    best_t, best_fb = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (proba >= t).astype(int)
        fb   = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if fb > best_fb:
            best_fb, best_t = fb, t
    return best_t


def _impute_with_train_mean(
    X: np.ndarray, train_means: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    if train_means is None:
        train_means = np.where(
            np.all(~np.isfinite(X), axis=0),
            0.0,
            np.nanmean(X, axis=0),
        )
    X_imp            = X.copy()
    nan_mask         = ~np.isfinite(X_imp)
    X_imp[nan_mask]  = np.take(train_means, np.where(nan_mask)[1])
    return X_imp, train_means


def save_spectrum_csv(
    waves   : dict,
    sample_id: str,
    cfg     : Config,
    out_dir : Path,
) -> None:
    """
    Save a multi-orientation spectrum CSV identical in structure to part 1,
    with columns: frequency_hz, <orient_1>, <orient_2>, <orient_3>.
    Orientations with no wave data are filled with NaN.
    """
    window   = Window(name=cfg.window_name)
    spectra  = {}
    freqs    = None
    for orient in cfg.orientations:
        wave = waves.get(orient)
        if wave is None:
            spectra[orient] = None
            continue
        amp  = AmplitudeSpectrum.from_wave(wave, window)
        if freqs is None:
            freqs = amp.frequencies
        spectra[orient] = amp.values

    if freqs is None:
        return

    n       = len(freqs)
    columns = [freqs] + [
        spectra[o] if spectra[o] is not None else np.full(n, np.nan)
        for o in cfg.orientations
    ]
    header  = "frequency_hz," + ",".join(cfg.orientations)
    data    = np.column_stack(columns)
    path    = out_dir / f"{sample_id}_spectrum.csv"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    log.info("  Spectrum CSV saved → %s", path)


def save_results_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    log.info("Results saved → %s", path)


def save_test_table_md(results_data: dict, output_path: Path) -> None:
    results = sorted(results_data["results"], key=lambda r: r["score"], reverse=True)
    lines   = [
        "| Sample | Asset type | RPM | Score | Prediction |",
        "|--------|------------|-----|-------|------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['sample_id']} "
            f"| {r['asset_type']} "
            f"| {r['rpm']:.0f} "
            f"| {r['score']:.6f} "
            f"| {r['prediction']} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Test results table saved → %s", output_path)


def run_extract(
    cfg        : Config,
    train_dir  : Path,
    train_meta : Path,
    output_dir : Path,
) -> None:
    log.info("Feature extraction & selection")

    meta = pd.read_csv(train_meta)
    meta = meta[meta["condition"].isin(["healthy", "structural_looseness"])].copy()
    log.info("Samples: %d  |  conditions: %s", len(meta), meta["condition"].value_counts().to_dict())

    rows = []
    for _, mrow in meta.iterrows():
        sid      = mrow["sample_id"]
        rpm      = float(mrow["rpm"])
        cond     = mrow["condition"]
        csv_path = train_dir / f"{sid}.csv"
        if not csv_path.is_file():
            log.warning("  SKIP (not found): %s", csv_path)
            continue
        orient_meta = _parse_orientation(mrow["orientation"])
        waves       = _load_train_waves(csv_path, orient_meta, cfg)
        feat_row    = extract_features_from_trio(
            waves.get("horizontal"), waves.get("vertical"), waves.get("axial"), rpm, cfg
        )
        feat_row["sample_id"] = sid
        feat_row["condition"] = cond
        rows.append(feat_row)

    df = pd.DataFrame(rows)
    log.info("Feature matrix: %d samples × %d features", len(df), len(cfg.feature_list))

    y = (df["condition"] == "structural_looseness").astype(int).values

    auroc_vals = {}
    for feat in cfg.feature_list:
        col  = df[feat].values
        mask = np.isfinite(col)
        if mask.sum() < 2:
            auroc_vals[feat] = 0.5
            continue
        try:
            a = roc_auc_score(y[mask], col[mask])
            a = max(a, 1.0 - a)
        except Exception:
            a = 0.5
        auroc_vals[feat] = round(float(a), 6)

    X_feats     = df[cfg.feature_list].values.astype(float)
    valid_mask  = np.all(np.isfinite(X_feats), axis=0)
    valid_feats = [f for f, v in zip(cfg.feature_list, valid_mask) if v]

    if len(valid_feats) >= 2:
        spear_mat, _ = spearmanr(df[valid_feats].values)
        if len(valid_feats) == 2:
            spear_mat = np.array([[1.0, spear_mat], [spear_mat, 1.0]])
    else:
        spear_mat = np.array([[1.0]])

    spearman_dict = {
        fi: {fj: round(float(spear_mat[i, j]), 6) for j, fj in enumerate(valid_feats)}
        for i, fi in enumerate(valid_feats)
    }

    healthy_df     = df[df["condition"] == "healthy"]
    loose_df       = df[df["condition"] == "structural_looseness"]
    fischer_scores = {}
    for feat in cfg.feature_list:
        h     = healthy_df[feat].dropna().values
        l     = loose_df[feat].dropna().values
        if len(h) < 2 or len(l) < 2:
            fischer_scores[feat] = 0.0
            continue
        denom = float(np.var(h) + np.var(l))
        fischer_scores[feat] = (
            round(float((np.mean(l) - np.mean(h)) ** 2 / denom), 6) if denom > 0 else 0.0
        )

    feature_stats = {}
    for feat in cfg.feature_list:
        stat = {}
        for cond_label in ["healthy", "structural_looseness"]:
            vals = df[df["condition"] == cond_label][feat].dropna().tolist()
            stat[cond_label] = [float(v) for v in vals]
        feature_stats[feat] = stat

    selected = list(cfg.feature_list)

    output = {
        "all_features"     : cfg.feature_list,
        "selected_features": selected,
        "auroc"            : auroc_vals,
        "spearman"         : spearman_dict,
        "fischer"          : fischer_scores,
        "feature_stats"    : feature_stats,
        "samples"          : [
            {
                "sample_id": str(r["sample_id"]),
                "condition": str(r["condition"]),
                **{f: (None if np.isnan(r[f]) else round(float(r[f]), 8)) for f in cfg.feature_list},
            }
            for _, r in df.iterrows()
        ],
    }
    save_results_json(output, output_dir / "features.json")

    for feat in cfg.feature_list:
        log.info(
            "  %-55s AUROC=%.4f  Fischer=%.4f",
            feat, auroc_vals[feat], fischer_scores[feat],
        )


def run_generalize(
    cfg        : Config,
    output_dir : Path,
) -> None:
    log.info("Generalization estimation (nested CV)")

    features_path = output_dir / "features.json"
    if not features_path.is_file():
        raise FileNotFoundError(f"Run extraction first. Expected: {features_path}")

    with open(features_path) as fh:
        feat_data = json.load(fh)

    feature_names = feat_data["selected_features"]
    df            = pd.DataFrame(feat_data["samples"])
    df            = df[df["condition"].isin(["healthy", "structural_looseness"])].copy()
    df_clean      = df[["condition"] + feature_names].dropna()
    y             = (df_clean["condition"] == "structural_looseness").astype(int).values
    X_raw         = df_clean[feature_names].values.astype(float)

    log.info(
        "Estimating generalization on %d samples | %d features | loose=%d healthy=%d",
        len(df_clean), len(feature_names), int(y.sum()), int((y == 0).sum()),
    )

    outer_skf    = StratifiedKFold(n_splits=cfg.n_outer, shuffle=True, random_state=cfg.random_state)
    inner_skf    = StratifiedKFold(n_splits=cfg.n_inner, shuffle=True, random_state=cfg.random_state)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(outer_skf.split(X_raw, y), 1):
        X_tr_raw, X_val_raw = X_raw[tr_idx], X_raw[val_idx]
        y_tr, y_val         = y[tr_idx], y[val_idx]
        X_tr, train_means   = _impute_with_train_mean(X_tr_raw)
        X_val, _            = _impute_with_train_mean(X_val_raw, train_means)

        rs = RandomizedSearchCV(
            _make_pipe(cfg.random_state), _param_dist(),
            n_iter=cfg.n_iter, cv=inner_skf,
            scoring="roc_auc", n_jobs=-1,
            random_state=cfg.random_state, refit=True,
        )
        rs.fit(X_tr, y_tr)
        best_params = rs.best_params_

        oof_proba = cross_val_predict(
            rs.best_estimator_, X_tr, y_tr,
            cv=inner_skf, method="predict_proba", n_jobs=-1,
        )[:, 1]
        threshold = _best_threshold(y_tr, oof_proba, cfg.f_beta)

        proba_val = rs.predict_proba(X_val)[:, 1]
        pred_val  = (proba_val >= threshold).astype(int)

        try:
            auroc = float(roc_auc_score(y_val, proba_val))
        except Exception:
            auroc = float("nan")

        cm = confusion_matrix(y_val, pred_val, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        result = {
            "fold"        : fold,
            "n_train"     : len(y_tr),
            "n_val"       : len(y_val),
            "auroc"       : round(auroc, 4),
            "f1"          : round(float(f1_score(y_val, pred_val, zero_division=0)), 4),
            "precision"   : round(float(precision_score(y_val, pred_val, zero_division=0)), 4),
            "recall"      : round(float(recall_score(y_val, pred_val, zero_division=0)), 4),
            "threshold"   : round(threshold, 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "best_C"      : float(best_params["lr__C"]),
            "best_penalty": best_params["lr__penalty"],
        }
        fold_results.append(result)
        log.info(
            "  Fold %d: AUROC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  Thr=%.3f  C=%.4f  pen=%s",
            fold, auroc, result["f1"], result["precision"], result["recall"],
            threshold, result["best_C"], result["best_penalty"],
        )

    avg = {m: round(float(np.mean([r[m] for r in fold_results])), 4) for m in ["auroc", "f1", "precision", "recall"]}
    std = {m: round(float(np.std([r[m]  for r in fold_results])), 4) for m in ["auroc", "f1", "precision", "recall"]}

    output = {
        "note"        : (
            "Unbiased generalization estimates. "
            "Describes pipeline performance, not the final model."
        ),
        "n_outer"     : cfg.n_outer,
        "n_inner"     : cfg.n_inner,
        "n_iter"      : cfg.n_iter,
        "f_beta"      : cfg.f_beta,
        "features"    : feature_names,
        "fold_results": fold_results,
        "cv_mean"     : avg,
        "cv_std"      : std,
    }
    save_results_json(output, output_dir / "generalization_metrics.json")

    log.info("Estimated performance (mean ± std):")
    for m in ["auroc", "f1", "precision", "recall"]:
        log.info("  %-12s %.4f ± %.4f", m, avg[m], std[m])

    save_cv_table_md(output, output_dir / "tables" / "cv_metrics_table.md")


def run_train(
    cfg        : Config,
    output_dir : Path,
) -> None:
    log.info("Final model training (full dataset)")

    features_path = output_dir / "features.json"
    if not features_path.is_file():
        raise FileNotFoundError(f"Run extraction first. Expected: {features_path}")

    with open(features_path) as fh:
        feat_data = json.load(fh)

    feature_names  = feat_data["selected_features"]
    df             = pd.DataFrame(feat_data["samples"])
    df             = df[df["condition"].isin(["healthy", "structural_looseness"])].copy()
    df_clean       = df[["condition"] + feature_names].dropna()
    y              = (df_clean["condition"] == "structural_looseness").astype(int).values
    X_raw          = df_clean[feature_names].values.astype(float)
    X, train_means = _impute_with_train_mean(X_raw)

    log.info(
        "Training on %d samples | %d features | loose=%d healthy=%d",
        len(df_clean), len(feature_names), int(y.sum()), int((y == 0).sum()),
    )

    inner_skf = StratifiedKFold(n_splits=cfg.n_inner, shuffle=True, random_state=cfg.random_state)
    rs        = RandomizedSearchCV(
        _make_pipe(cfg.random_state), _param_dist(),
        n_iter=cfg.n_iter, cv=inner_skf,
        scoring="roc_auc", n_jobs=-1,
        random_state=cfg.random_state, refit=True,
    )
    rs.fit(X, y)
    best_params = rs.best_params_
    final_pipe  = rs.best_estimator_
    log.info("Best hyperparameters: C=%.4f  penalty=%s", best_params["lr__C"], best_params["lr__penalty"])

    oof_proba = cross_val_predict(
        final_pipe, X, y, cv=inner_skf, method="predict_proba", n_jobs=-1,
    )[:, 1]
    threshold = _best_threshold(y, oof_proba, cfg.f_beta)
    log.info("Threshold (from OOF predictions): %.4f", threshold)

    lr     = final_pipe.named_steps["lr"]
    scaler = final_pipe.named_steps["scaler"]

    model_bundle = {
        "pipeline"   : final_pipe,
        "scaler"     : scaler,
        "lr"         : lr,
        "features"   : feature_names,
        "train_means": train_means,
        "threshold"  : threshold,
    }
    model_path = output_dir / "model.joblib"
    joblib.dump(model_bundle, model_path)
    log.info("Model saved → %s", model_path)

    train_info = {
        "note"       : "No performance metrics here. See generalization_metrics.json for honest estimates.",
        "n_samples"  : len(df_clean),
        "features"   : feature_names,
        "best_C"     : round(float(best_params["lr__C"]), 6),
        "best_penalty": best_params["lr__penalty"],
        "threshold"  : round(threshold, 4),
        "coefs"      : {
            feat: round(float(c), 6)
            for feat, c in zip(feature_names, lr.coef_.flatten())
        },
        "intercept"  : round(float(lr.intercept_[0]), 6),
    }
    save_results_json(train_info, output_dir / "training_info.json")


def run_infer(
    cfg       : Config,
    test_dir  : Path,
    test_meta : Path,
    output_dir: Path,
) -> None:
    log.info("Inference on test data")

    model_path = output_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Run training first. Expected: {model_path}")

    model_bundle = joblib.load(model_path)
    log.info(
        "Model loaded | features=%d  threshold=%.3f",
        len(model_bundle["features"]), model_bundle["threshold"],
    )

    meta = pd.read_csv(test_meta)
    if "asset" in meta.columns and "asset_type" not in meta.columns:
        meta = meta.rename(columns={"asset": "asset_type"})

    spectra_dir = output_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for _, mrow in meta.iterrows():
        sid        = str(mrow["sample_id"])
        rpm        = float(mrow["rpm"])
        asset_type = mrow.get("asset_type", "unknown")
        csv_path   = test_dir / f"{sid}.csv"
        if not csv_path.is_file():
            log.warning("  SKIP (not found): %s", csv_path)
            continue

        orient_meta = _parse_orientation(mrow["orientation"])
        waves       = _load_test_waves(csv_path, orient_meta, cfg)

        feat_row = extract_features_from_trio(
            waves.get("horizontal"), waves.get("vertical"), waves.get("axial"), rpm, cfg
        )

        features    = model_bundle["features"]
        pipeline    = model_bundle["pipeline"]
        train_means = model_bundle["train_means"]
        threshold   = model_bundle["threshold"]

        x        = np.array([feat_row.get(f, np.nan) for f in features], dtype=float).reshape(1, -1)
        x, _     = _impute_with_train_mean(x, train_means)
        score    = float(pipeline.predict_proba(x)[0, 1])
        label    = "structural_looseness" if score >= threshold else "healthy"

        lr      = model_bundle["lr"]
        scaler  = model_bundle["scaler"]
        x_sc    = scaler.transform(x).flatten()
        coefs   = lr.coef_.flatten()
        contribs = sorted(
            [
                {
                    "feature"     : feat,
                    "contribution": round(float(c), 6),
                    "scaled_value": round(float(xv), 6),
                    "coef"        : round(float(coef), 6),
                    "raw_value"   : (
                        round(float(feat_row[feat]), 8)
                        if feat in feat_row and np.isfinite(feat_row.get(feat, np.nan))
                        else None
                    ),
                }
                for feat, c, xv, coef in zip(features, x_sc * coefs, x_sc, coefs)
            ],
            key=lambda d: abs(d["contribution"]),
            reverse=True,
        )

        save_spectrum_csv(waves, sid, cfg, spectra_dir)

        results.append({
            "sample_id"    : sid,
            "asset_type"   : str(asset_type),
            "rpm"          : float(rpm),
            "score"        : round(score, 6),
            "prediction"   : label,
            "is_loose"     : label == "structural_looseness",
            "features"     : {
                f: (None if np.isnan(feat_row.get(f, np.nan)) else round(float(feat_row[f]), 8))
                for f in cfg.feature_list
            },
            "contributions": contribs,
        })

        flag = "structural_looseness" if label == "structural_looseness" else "healthy"
        log.info("  %s  %-12s  score=%.4f  %s", sid[:12], asset_type, score, flag)

    n_loose   = sum(1 for r in results if r["is_loose"])
    n_healthy = len(results) - n_loose

    output = {
        "threshold": model_bundle["threshold"],
        "n_loose"  : n_loose,
        "n_healthy": n_healthy,
        "results"  : results,
    }
    save_results_json(output, output_dir / "test_results.json")
    save_test_table_md(output, output_dir / "tables" / "test_results_table.md")
    log.info("Done. %d loose / %d healthy.", n_loose, n_healthy)


def save_cv_table_md(metrics_data: dict, output_path: Path) -> None:
    fold_results = metrics_data["fold_results"]
    cv_mean      = metrics_data.get("cv_mean", {})
    cv_std       = metrics_data.get("cv_std",  {})
    lines        = [
        "| Fold | AUROC | F1 | Precision | Recall |",
        "|------|-------|----|-----------|--------|",
    ]
    for r in fold_results:
        lines.append(
            f"| Fold {r['fold']} "
            f"| {r['auroc']:.4f} "
            f"| {r['f1']:.4f} "
            f"| {r['precision']:.4f} "
            f"| {r['recall']:.4f} |"
        )
    lines.append(
        f"| **Mean ± Std** "
        f"| {cv_mean.get('auroc', 0):.4f} ± {cv_std.get('auroc', 0):.4f} "
        f"| {cv_mean.get('f1', 0):.4f} ± {cv_std.get('f1', 0):.4f} "
        f"| {cv_mean.get('precision', 0):.4f} ± {cv_std.get('precision', 0):.4f} "
        f"| {cv_mean.get('recall', 0):.4f} ± {cv_std.get('recall', 0):.4f} |"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("CV metrics table saved → %s", output_path)


class LoosenessModel:
    """
    Looseness detector.

    Wraps the trained Logistic Regression model.
    """

    def __init__(self, model_path: str | Path = DEFAULT_OUTPUT / "model.joblib",
                 rpm: float = 1500.0, cfg: Optional[Config] = None, **params) -> None:
        self.params      = params
        self.rpm         = rpm
        self.cfg         = cfg or Config()
        self._bundle     = None
        self._model_path = Path(model_path)

    def _load(self) -> None:
        if self._bundle is None:
            if not self._model_path.is_file():
                raise FileNotFoundError(
                    f"Model not found at {self._model_path}. Run training first.")
            self._bundle = joblib.load(self._model_path)

    def _feats(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> dict:
        self._load()
        return extract_features_from_trio(wave_hor, wave_ver, wave_axi, self.rpm, self.cfg)

    def predict(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> bool:
        feat_row    = self._feats(wave_hor, wave_ver, wave_axi)
        features    = self._bundle["features"]
        pipeline    = self._bundle["pipeline"]
        train_means = self._bundle["train_means"]
        threshold   = self._bundle["threshold"]
        x           = np.array([feat_row.get(f, np.nan) for f in features], dtype=float).reshape(1, -1)
        x, _        = _impute_with_train_mean(x, train_means)
        score       = float(pipeline.predict_proba(x)[0, 1])
        return score >= threshold

    def score(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> float:
        feat_row    = self._feats(wave_hor, wave_ver, wave_axi)
        features    = self._bundle["features"]
        pipeline    = self._bundle["pipeline"]
        train_means = self._bundle["train_means"]
        x           = np.array([feat_row.get(f, np.nan) for f in features], dtype=float).reshape(1, -1)
        x, _        = _impute_with_train_mean(x, train_means)
        return float(pipeline.predict_proba(x)[0, 1])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feature extraction → generalization estimation → "
            "model training → inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     "-c", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-dir",        type=str, default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--test-dir",         type=str, default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--train-meta",       type=str, default=str(DEFAULT_TRAIN_META))
    parser.add_argument("--test-meta",        type=str, default=str(DEFAULT_TEST_META))
    parser.add_argument("--output",     "-o", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--run", type=str,
        choices=["extract", "generalize", "train", "infer"],
        default=None,
        help="Run a single step (default: run all).",
    )
    return parser


def main() -> None:
    args       = _build_arg_parser().parse_args()
    cfg        = Config.from_json(args.config)
    train_dir  = Path(args.train_dir)
    test_dir   = Path(args.test_dir)
    train_meta = Path(args.train_meta)
    test_meta  = Path(args.test_meta)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = args.run

    if run is None or run == "extract":
        run_extract(cfg, train_dir, train_meta, output_dir)
    if run is None or run == "generalize":
        run_generalize(cfg, output_dir)
    if run is None or run == "train":
        run_train(cfg, output_dir)
    if run is None or run == "infer":
        run_infer(cfg, test_dir, test_meta, output_dir)

    log.info("Done. All outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
