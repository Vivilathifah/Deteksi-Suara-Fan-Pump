"""Konfigurasi & fungsi bersama untuk seluruh pipeline MIMII DUE."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW, EXTRACTED, PROCESSED, META = DATA / "raw", DATA / "extracted", DATA / "processed", DATA / "metadata"
FEATURES, DISTANCES, CLUSTERING, RESULTS = ROOT / "features", ROOT / "distances", ROOT / "clustering", ROOT / "results"

MANIFEST = META / "audio_manifest.csv"
PROC_MANIFEST = META / "processed_manifest.csv"

MACHINES = ["fan", "pump"]
FEATURE_TYPES = ["MFCC", "LogMel"]
DISTANCE_TYPES = ["DTW", "FastDTW"]

# Parameter audio & fitur
SR = 16000
N_FFT, HOP, N_MELS, N_MFCC = 1024, 512, 64, 20

# Zenodo: dev = record 4740355, eval (train) = record 4660992  (cek ulang jika link berubah)
ZIPS = {
    "dev_data_fan.zip": "https://zenodo.org/records/4740355/files/dev_data_fan.zip?download=1",
    "dev_data_pump.zip": "https://zenodo.org/records/4740355/files/dev_data_pump.zip?download=1",
    "eval_data_fan_train.zip": "https://zenodo.org/records/4660992/files/eval_data_fan_train.zip?download=1",
    "eval_data_pump_train.zip": "https://zenodo.org/records/4660992/files/eval_data_pump_train.zip?download=1",
}

FNAME_RE = re.compile(
    r"section_(?P<section>\d+)_(?P<domain>source|target)_(?P<split>train|test)_"
    r"(?P<condition>normal|anomaly)_(?P<idx>\d+)"
)


def parse_filename(name: str) -> dict:
    m = FNAME_RE.search(name)
    if not m:
        return dict(section=-1, domain="unknown", split="unknown", condition="unknown", idx=-1)
    d = m.groupdict()
    d["section"], d["idx"] = int(d["section"]), int(d["idx"])
    return d


def ensure_dirs():
    for p in [RAW, EXTRACTED, PROCESSED, META, FEATURES, DISTANCES, CLUSTERING / "membership",
              CLUSTERING / "labels", CLUSTERING / "models", RESULTS / "metrics", RESULTS / "plots", RESULTS / "tables"]:
        p.mkdir(parents=True, exist_ok=True)


def tag(machine, section, feature, dist):
    return f"{machine}_sec{int(section):02d}_{feature}_{dist}"


def select_group(pm: pd.DataFrame, machine: str, section: int, max_n: int | None, seed: int = 42) -> pd.DataFrame:
    """Pilih sampel deterministik (stratified domain x kondisi) -> DTW & FastDTW memakai sampel identik."""
    g = pm[(pm.machine == machine) & (pm.section == section)].sort_values("file_id").reset_index(drop=True)
    if not max_n or len(g) <= max_n:
        return g
    frac = max_n / len(g)
    parts = [s.sample(n=max(1, round(len(s) * frac)), random_state=seed) for _, s in g.groupby(["domain", "condition"])]
    out = pd.concat(parts)
    if len(out) > max_n:
        out = out.sample(n=max_n, random_state=seed)
    return out.sort_values("file_id").reset_index(drop=True)


def load_features(feature: str, machine: str, ids) -> np.ndarray:
    """Muat fitur (N, T, D), distandardisasi z-score per koefisien pada grup tersebut."""
    arrs = [np.load(FEATURES / feature / machine / f"{i}.npy").T for i in ids]  # (T, D)
    X = np.stack(arrs).astype(np.float64)
    mu, sd = X.mean(axis=(0, 1), keepdims=True), X.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X - mu) / sd


def list_groups(pm: pd.DataFrame):
    return sorted({(m, int(s)) for m, s in zip(pm.machine, pm.section) if s >= 0})
