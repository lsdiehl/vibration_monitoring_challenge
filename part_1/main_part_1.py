# -*- coding: utf-8 -*-
"""
Part 1 — Carpet region detection.

Processes all CSV files in the input directory, detects carpet regions in the
frequency spectrum, and saves per-sample results.

Outputs (per sample)
--------------------
  results/<stem>_spectrum.csv   — frequency_hz, amplitude
  results/<stem>_results.json   — metadata, CPR metric, detected regions
  results/tables/cpr_table.md   — all samples ranked by CPR

Usage
-----
    python main_part_1.py
    python main_part_1.py --config config_part_1.json --input data/ --output results/
    python main_part_1.py --psd-threshold-db -70
"""

from __future__ import annotations

import argparse
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from scipy.signal import medfilt, welch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config_part_1.json"
DEFAULT_INPUT  = Path(__file__).parent / "data"
DEFAULT_OUTPUT = Path(__file__).parent / "results"

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


class PSD(Spectrum):
    def unit_label(self) -> str:
        return f"PSD [{self.unit}²/Hz]"

    @classmethod
    def from_wave(
        cls,
        wave: "Wave",
        window: Window,
        ratio_delta_f: float,
        overlap: float,
    ) -> "PSD":
        sig      = np.asarray(wave.signal, dtype=float)
        N        = len(sig)
        fs       = wave.sample_rate
        delta_f  = ratio_delta_f * (fs / N)
        nperseg  = int(fs / delta_f)
        noverlap = int(nperseg * overlap)
        window.resize(nperseg)
        f, p = welch(
            sig,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            window=window.array,
            return_onesided=True,
        )
        return cls(frequencies=f, values=p, window=window, unit=wave.unit)


class CarpetRegion(BaseModel):
    start_hz  : float = Field(..., description="Start frequency in Hz")
    end_hz    : float = Field(..., description="End frequency in Hz")
    id        : int   = Field(..., description="1-based index ordered by ascending start frequency")
    power     : float = Field(..., description="Integrated PSD power within this band [signal unit²]")

    model_config = {"frozen": True}

    @computed_field
    @property
    def bandwidth_hz(self) -> float:
        return self.end_hz - self.start_hz


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

    @classmethod
    def from_csv(cls, path: Path, quantity: str = "acceleration", unit: str = "g") -> "Wave":
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got '{path.suffix}' ('{path.name}').")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, newline="") as fh:
            header = [h.strip().lower() for h in fh.readline().strip().split(",")]

        missing = {"t", "data"} - set(header)
        if missing:
            raise ValueError(
                f"CSV '{path.name}' is missing required columns: {missing}. Found: {header}"
            )

        col_t    = header.index("t")
        col_data = header.index("data")

        raw = np.genfromtxt(path, delimiter=",", skip_header=1)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        if raw.shape[0] == 0:
            raise ValueError(f"CSV '{path.name}' contains no data rows.")

        return cls(
            time     = raw[:, col_t].tolist(),
            signal   = raw[:, col_data].tolist(),
            quantity = quantity,
            unit     = unit,
        )


class Config(BaseModel):
    wave_quantity     : Literal["acceleration", "velocity", "displacement"] = Field(
        "acceleration", description="Default physical quantity for loaded waves"
    )
    wave_unit         : str              = Field("g",       description="Default measurement unit")
    median_window_hz  : float            = Field(100.0,    description="Median filter width [Hz]")
    psd_threshold_db  : float            = Field(-80.0,    description="Detection threshold [dB] — must be < 0")
    min_bandwidth_hz  : float            = Field(100.0,    description="Minimum band width [Hz]")
    min_start_freq_hz : float            = Field(1000.0,   description="Minimum band start frequency [Hz]")
    max_band_gap_hz   : float            = Field(150.0,    description="Maximum merging gap [Hz]")
    ratio_delta_f     : float            = Field(1.1,      description="Welch segment multiplier — must be > 1")
    overlap           : float            = Field(0.9,      description="Welch overlap fraction [0, 1)")
    window_name       : Literal["hanning"] = Field("hanning", description="FFT window type")
    
    model_config = {"extra": "forbid"}

    @field_validator("psd_threshold_db")
    @classmethod
    def _threshold_must_be_negative(cls, v: float) -> float:
        if v >= 0.0:
            raise ValueError(f"'psd_threshold_db' must be negative, got {v}.")
        return v

    @field_validator("ratio_delta_f")
    @classmethod
    def _ratio_must_exceed_one(cls, v: float) -> float:
        if v <= 1.0:
            raise ValueError(f"'ratio_delta_f' must be > 1.0, got {v}.")
        return v

    @field_validator("overlap")
    @classmethod
    def _overlap_range(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"'overlap' must be in [0, 1), got {v}.")
        return v

    @field_validator("max_band_gap_hz")
    @classmethod
    def _gap_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"'max_band_gap_hz' must be ≥ 0, got {v}.")
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

    def to_model_params(self) -> Dict[str, Any]:
        return {
            "median_window_hz" : self.median_window_hz,
            "psd_threshold_db" : self.psd_threshold_db,
            "min_bandwidth_hz" : self.min_bandwidth_hz,
            "min_start_freq_hz": self.min_start_freq_hz,
            "max_band_gap_hz"  : self.max_band_gap_hz,
            "ratio_delta_f"    : self.ratio_delta_f,
            "overlap"          : self.overlap,
            "window_name"      : self.window_name,
        }

class Model:
    """
    Detects carpet regions from a Wave's PSD.

    Parameters (via **params or Config.to_model_params())
    ------------------------------------------------------
    median_window_hz  : median filter width [Hz]
    psd_threshold_db  : detection threshold [dB] — must be negative
    min_bandwidth_hz  : minimum band width [Hz]
    min_start_freq_hz : ignore bands below this frequency [Hz]
    max_band_gap_hz   : merge bands whose gap is smaller than this [Hz]
    ratio_delta_f     : Welch segment length multiplier — must be > 1
    overlap           : Welch segment overlap fraction [0, 1)
    window_name       : FFT window type
    """

    def __init__(self, **params) -> None:
        self.params            = params
        self.median_window_hz  = float(params.get("median_window_hz",  100.0))
        self.psd_threshold_db  = float(params.get("psd_threshold_db",  -80.0))
        self.min_bandwidth_hz  = float(params.get("min_bandwidth_hz",  100.0))
        self.min_start_freq_hz = float(params.get("min_start_freq_hz", 1000.0))
        self.max_band_gap_hz   = float(params.get("max_band_gap_hz",   150.0))
        self.ratio_delta_f     = float(params.get("ratio_delta_f",     1.1))
        self.overlap           = float(params.get("overlap",           0.9))
        self.window_name       = str(params.get("window_name",         "hanning"))
        self._validate_params()

    def _validate_params(self) -> None:
        if self.psd_threshold_db >= 0.0:
            raise ValueError(f"'psd_threshold_db' must be negative, got {self.psd_threshold_db}.")
        if self.ratio_delta_f <= 1.0:
            raise ValueError(f"'ratio_delta_f' must be > 1.0, got {self.ratio_delta_f}.")
        if not (0.0 <= self.overlap < 1.0):
            raise ValueError(f"'overlap' must be in [0, 1), got {self.overlap}.")
        if self.max_band_gap_hz < 0.0:
            raise ValueError(f"'max_band_gap_hz' must be ≥ 0, got {self.max_band_gap_hz}.")

    def predict(self, wave: Wave) -> List[CarpetRegion]:
        self._validate_against_wave(wave)
        window       = Window(name=self.window_name)
        psd          = PSD.from_wave(wave, window, self.ratio_delta_f, self.overlap)
        psd_smooth   = self._smooth_median(psd.values, psd.frequencies)
        psd_smooth_db = self._to_db(psd_smooth, psd.frequencies)
        raw_bands    = self._find_bands(psd.frequencies, psd_smooth_db)
        merged       = self._merge_bands(raw_bands)
        regions      = self._build_regions(psd.frequencies, psd.values, merged)
        log.info("  Detected %d carpet region(s).", len(regions))
        for r in regions:
            log.info(
                "    [id=%d]  %.1f Hz – %.1f Hz  bw=%.1f Hz  power=%.4e %s²",
                r.id, r.start_hz, r.end_hz, r.bandwidth_hz, r.power, psd.unit,
            )
        return regions

    def cpr(self, regions: List[CarpetRegion], wave: Wave) -> float:
        """Carpet Power Ratio: total_power_within_carpets / total_power_above_min_start_freq."""
        window      = Window(name=self.window_name)
        psd         = PSD.from_wave(wave, window, self.ratio_delta_f, self.overlap)
        band_power  = sum(r.power for r in regions)
        ref_mask    = psd.frequencies >= self.min_start_freq_hz
        total_power = np.trapezoid(psd.values[ref_mask], psd.frequencies[ref_mask])
        return float(band_power / total_power) if total_power > 0 else 0.0

    def _validate_against_wave(self, wave: Wave) -> None:
        nyquist     = wave.sample_rate / 2.0
        max_allowed = nyquist - self.min_start_freq_hz
        if self.min_bandwidth_hz >= max_allowed:
            raise ValueError(
                f"'min_bandwidth_hz' ({self.min_bandwidth_hz} Hz) must be smaller than "
                f"nyquist − min_start_freq_hz ({nyquist:.1f} − {self.min_start_freq_hz} = {max_allowed:.1f} Hz)."
            )

    def _smooth_median(self, y: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        N  = len(y)
        df = (freqs[-1] - freqs[0]) / (N - 1) if N > 1 else 1.0
        ks = int(round(self.median_window_hz / df))
        ks = max(1, min(ks, N))
        ks = ks if ks % 2 == 1 else ks - 1
        return medfilt(y, kernel_size=ks)

    def _to_db(self, psd_smooth: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        mask    = freqs >= self.min_start_freq_hz
        psd_ref = np.nanmax(psd_smooth[mask])
        safe    = np.where(psd_smooth > 0, psd_smooth, np.nan)
        return 10.0 * np.log10(safe / psd_ref)

    def _find_bands(self, freqs: np.ndarray, spectrum_db: np.ndarray):
        above   = np.where(~np.isnan(spectrum_db), spectrum_db >= self.psd_threshold_db, False)
        bands   = []
        in_band = False
        i_start = 0
        for i, flag in enumerate(above):
            if flag and not in_band:
                in_band = True
                i_start = i
            elif not flag and in_band:
                in_band = False
                f_lo, f_hi = freqs[i_start], freqs[i]
                if f_lo >= self.min_start_freq_hz and (f_hi - f_lo) >= self.min_bandwidth_hz:
                    bands.append((f_lo, f_hi))
        if in_band:
            f_lo, f_hi = freqs[i_start], freqs[-1]
            if f_lo >= self.min_start_freq_hz and (f_hi - f_lo) >= self.min_bandwidth_hz:
                bands.append((f_lo, f_hi))
        return bands

    def _merge_bands(self, bands):
        if not bands or self.max_band_gap_hz == 0:
            return bands
        merged = [bands[0]]
        for f_lo, f_hi in bands[1:]:
            prev_lo, prev_hi = merged[-1]
            if (f_lo - prev_hi) < self.max_band_gap_hz:
                merged[-1] = (prev_lo, max(prev_hi, f_hi))
            else:
                merged.append((f_lo, f_hi))
        return merged

    def _build_regions(self, freqs: np.ndarray, psd_values: np.ndarray, bands) -> List[CarpetRegion]:
        regions = []
        for idx, (f_lo, f_hi) in enumerate(bands, start=1):
            mask  = (freqs >= f_lo) & (freqs <= f_hi)
            power = (
                float(np.trapezoid(psd_values[mask], freqs[mask]))
                if mask.sum() >= 2 else 0.0
            )
            regions.append(CarpetRegion(id=idx, start_hz=f_lo, end_hz=f_hi, power=power))
        return regions


def save_spectrum_csv(amp_spectrum: AmplitudeSpectrum, path: Path) -> None:
    header = "frequency_hz,amplitude"
    data   = np.column_stack([amp_spectrum.frequencies, amp_spectrum.values])
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    log.info("  Spectrum CSV saved → %s", path)


def save_results_json(
    wave         : Wave,
    regions      : List[CarpetRegion],
    cpr          : float,
    amp_spectrum : AmplitudeSpectrum,
    source_name  : str,
    path         : Path,
) -> None:
    result = {
        "source_file"        : source_name,
        "quantity"           : wave.quantity,
        "unit"               : wave.unit,
        "n_samples"          : len(wave.time),
        "duration_s"         : round(wave.duration, 6),
        "sample_rate_hz"     : round(wave.sample_rate, 4),
        "dt_s"               : round(wave.dt, 9),
        "freq_resolution_hz" : round(amp_spectrum.freq_resolution, 6),
        "nyquist_hz"         : round(amp_spectrum.nyquist_hz, 4),
        "carpet_power_ratio" : round(cpr, 8),
        "n_carpet_regions"   : len(regions),
        "carpet_regions"     : [
            {
                "id"          : r.id,
                "start_hz"    : round(r.start_hz, 4),
                "end_hz"      : round(r.end_hz, 4),
                "bandwidth_hz": round(r.bandwidth_hz, 4),
                "power"       : r.power,
            }
            for r in regions
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log.info("  Results JSON saved → %s", path)


def save_cpr_table_md(results_dir: Path, output_path: Path) -> None:
    rows = []
    for json_path in sorted(results_dir.glob("*_results.json")):
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        stem = Path(meta["source_file"]).stem
        rows.append((stem, meta["carpet_power_ratio"], meta["n_carpet_regions"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    lines = [
        "| Rank | Sample | CPR | Carpet regions |",
        "|------|--------|-----|----------------|",
    ]
    for rank, (sample, cpr, n_regions) in enumerate(rows, start=1):
        lines.append(f"| {rank} | {sample} | {cpr:.6f} | {n_regions} |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("CPR table saved → %s", output_path)
    

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect carpet regions in all CSV files in a folder (part 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--input",  "-i", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--quantity",          type=str,   default=None,
                        choices=["acceleration", "velocity", "displacement"])
    parser.add_argument("--unit",              type=str,   default=None)
    parser.add_argument("--median-window-hz",  type=float, default=None)
    parser.add_argument("--psd-threshold-db",  type=float, default=None)
    parser.add_argument("--min-bandwidth-hz",  type=float, default=None)
    parser.add_argument("--min-start-freq-hz", type=float, default=None)
    parser.add_argument("--max-band-gap-hz",   type=float, default=None)
    parser.add_argument("--ratio-delta-f",     type=float, default=None)
    parser.add_argument("--overlap",           type=float, default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg  = Config.from_json(args.config)

    overrides = {
        k: v for k, v in {
            "median_window_hz" : args.median_window_hz,
            "psd_threshold_db" : args.psd_threshold_db,
            "min_bandwidth_hz" : args.min_bandwidth_hz,
            "min_start_freq_hz": args.min_start_freq_hz,
            "max_band_gap_hz"  : args.max_band_gap_hz,
            "ratio_delta_f"    : args.ratio_delta_f,
            "overlap"          : args.overlap,
        }.items() if v is not None
    }
    if overrides:
        cfg = cfg.model_copy(update=overrides)
        log.info("CLI overrides applied: %s", overrides)

    quantity   = args.quantity or cfg.wave_quantity
    unit       = args.unit     or cfg.wave_unit
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        log.warning("No CSV files found in '%s'.", input_dir)
        return

    log.info("Found %d CSV file(s) in '%s'.", len(csv_files), input_dir)
    model = Model(**cfg.to_model_params())

    for csv_path in csv_files:
        log.info("Processing '%s' ...", csv_path.name)
        stem = csv_path.stem
        try:
            wave         = Wave.from_csv(csv_path, quantity=quantity, unit=unit)
            regions      = model.predict(wave)
            cpr_val      = model.cpr(regions, wave)
            log.info("  CPR: %.6f", cpr_val)
            window       = Window(name=model.window_name)
            amp_spectrum = AmplitudeSpectrum.from_wave(wave, window)
            save_spectrum_csv(amp_spectrum, output_dir / f"{stem}_spectrum.csv")
            save_results_json(
                wave, regions, cpr_val, amp_spectrum,
                source_name=csv_path.name,
                path=output_dir / f"{stem}_results.json",
            )
        except Exception as exc:
            log.error("  Failed to process '%s': %s", csv_path.name, exc)
            continue

    save_cpr_table_md(output_dir, output_dir / "tables" / "cpr_table.md")
    log.info("Done. Results saved to '%s'.", output_dir)


if __name__ == "__main__":
    main()
