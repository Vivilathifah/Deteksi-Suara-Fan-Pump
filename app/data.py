"""Pemuat data untuk dashboard Streamlit.

Prinsip supaya cepat:
  * hanya numpy + pandas yang di-import di sini (tanpa numba/librosa/sklearn);
  * setiap file dibaca sekali lalu disimpan di cache (dibaca ulang otomatis bila file berubah);
  * data berat (MDS, pratinjau fitur, waveform, klip audio) diambil dari dashboard_data/ hasil 14_pack_for_dashboard.py;
    bila paket belum ada, dipakai file asli (lebih lambat, hanya untuk dashboard lokal).
"""
import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import (CLUSTERING, DATA, DISTANCES, DISTANCE_TYPES, FEATURES, FEATURE_TYPES, HOP, MACHINES,  # noqa: E402,F401
                    MANIFEST, N_FFT, N_MELS, N_MFCC, PROC_MANIFEST, PROCESSED, RESULTS, SR, tag)

PACK = ROOT / "dashboard_data"
PRED = ROOT / "prediction"
PRED_RES = RESULTS / "prediction"
METHODS = ["DTW", "FastDTW", "Euclid"]
TASK_NAMES = {"condition_fan": "Normal vs anomali · Fan", "condition_pump": "Normal vs anomali · Pump",
              "machine": "Jenis mesin · Fan vs Pump"}


# ------------------------------------------------------------------ cache berbasis waktu modifikasi file
def _mtime(p):
    p = Path(p)
    return p.stat().st_mtime if p.exists() else None


@functools.lru_cache(maxsize=128)
def _csv(path, mtime):
    return None if mtime is None else pd.read_csv(path)


def csv(path):
    """CSV -> DataFrame (atau None bila belum ada). Jangan diubah in-place: pakai .copy()/.assign()."""
    return _csv(str(path), _mtime(path))


@functools.lru_cache(maxsize=64)
def _npz(path, mtime):
    if mtime is None:
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def npz(path):
    return _npz(str(path), _mtime(path))


@functools.lru_cache(maxsize=64)
def _npy(path, mtime):
    return None if mtime is None else np.load(path)


def npy(path):
    return _npy(str(path), _mtime(path))


@functools.lru_cache(maxsize=1)
def has_local_raw():
    """True bila WAV mentah ada di komputer ini (dashboard lokal). Online biasanya False."""
    return next((DATA / "extracted").glob("*/*/*.wav"), None) is not None


# ------------------------------------------------------------------ grup jarak (mesin x section x fitur)
def groups_available():
    """{(mesin, section, fitur)} yang punya matriks jarak (dari paket atau dari distances/)."""
    out = set()
    for p in PACK.glob("groups/*.npz"):
        machine, rest = p.stem.split("_", 1)
        feat, sec = rest.split("_sec")
        out.add((machine, int(sec), feat))
    for p in DISTANCES.glob("DTW/*/*_sec??.npy"):
        feat, sec = p.stem.split("_sec")
        out.add((p.parent.name, int(sec), feat))
    return out


def sections_for(machine, feat=None):
    return sorted({s for m, s, f in groups_available() if m == machine and (feat is None or f == feat)})


def _mds(D):
    from sklearn.manifold import MDS  # hanya dipakai bila paket belum dibuat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = MDS(2, dissimilarity="precomputed", random_state=0, n_init=1, init="random")
        except TypeError:
            m = MDS(2, metric="precomputed", random_state=0, n_init=1, init="random")
        return m.fit_transform(D).astype(np.float32)


@functools.lru_cache(maxsize=32)
def _group_fallback(machine, sec, feat, stamp):
    out = {}
    for dist in DISTANCE_TYPES:
        p = DISTANCES / dist / machine / f"{feat}_sec{sec:02d}.npy"
        if p.exists():
            out[dist] = np.load(p).astype(np.float32)
            out["ids"] = pd.read_csv(p.with_name(p.stem + "_ids.csv")).file_id.astype(str).values
    return out or None


def group(machine, sec, feat):
    """dict(ids, DTW, FastDTW, mds_DTW, mds_FastDTW) — kunci bisa hilang bila belum dihitung."""
    g = npz(PACK / "groups" / f"{machine}_{feat}_sec{int(sec):02d}.npz")
    if g is not None:
        return g
    stamp = _mtime(DISTANCES / "DTW" / machine / f"{feat}_sec{int(sec):02d}.npy")
    return _group_fallback(machine, int(sec), feat, stamp)


@functools.lru_cache(maxsize=32)
def _mds_cached(machine, sec, feat, dist, stamp):
    return _mds(group(machine, sec, feat)[dist])


def mds_coords(machine, sec, feat, dist):
    g = group(machine, sec, feat)
    if g is None or dist not in g:
        return None
    if f"mds_{dist}" in g:
        return g[f"mds_{dist}"]
    return _mds_cached(machine, int(sec), feat, dist, _mtime(DISTANCES / dist / machine / f"{feat}_sec{int(sec):02d}.npy"))


# ------------------------------------------------------------------ contoh audio & fitur
def samples():
    return csv(PACK / "samples.csv")


def file_choices(machine, need="audio", limit=500):
    """Pilihan file untuk halaman Audio/Fitur. Contoh yang dipaketkan (bertanda ★) selalu tersedia, juga online."""
    ch = {}
    s = samples()
    if s is not None:
        for fid in s[s.machine == machine].file_id:
            ch[fid] = f"★ {fid}"
    pm = csv(PROC_MANIFEST)
    if pm is not None:
        sub = pm[pm.machine == machine].file_id
        local = PROCESSED / machine if need == "audio" else FEATURES / FEATURE_TYPES[0] / machine
        if local.exists() and any(local.iterdir()):
            for fid in sub.head(limit):
                ch.setdefault(fid, fid)
    return ch


def shrink_frames(A, T=160):
    D, T0 = A.shape
    if T0 <= T:
        return A
    edges = np.linspace(0, T0, T + 1).astype(int)
    return np.add.reduceat(A, edges[:-1], axis=1) / np.diff(edges)[None, :]


def feature_preview(machine, fid, feat):
    """(matriks kecil (D, <=160), bentuk asli) atau (None, None)."""
    z = npz(PACK / f"features_{machine}.npz")
    if z is not None and feat in z:
        idx = np.where(z["ids"] == fid)[0]
        if len(idx):
            i = int(idx[0])
            shape = tuple(int(v) for v in z[f"{feat}_shape"][i])
            A = z[feat][i].astype(np.float32)
            A = A[:, ~np.isnan(A).all(0)]
            A = A[~np.isnan(A).all(1)]
            return A, shape
    p = FEATURES / feat / machine / f"{fid}.npy"
    if p.exists():
        A = npy(p)
        return shrink_frames(A).astype(np.float32), A.shape
    return None, None


def envelope_from_signal(y, bins=1200):
    y = np.asarray(y, dtype=np.float32)
    edges = np.linspace(0, len(y), min(bins, len(y)) + 1).astype(int)
    return np.stack([np.minimum.reduceat(y, edges[:-1]), np.maximum.reduceat(y, edges[:-1])])


@functools.lru_cache(maxsize=64)
def _wav_info(path, mtime):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=True, dtype="float32")
    y = y.mean(1)
    return envelope_from_signal(y), sr, len(y) / sr, float(np.abs(y).max())


def audio_view(machine, fid):
    """Untuk halaman Audio: dict kind -> (envelope, sr, durasi, puncak, url) untuk 'raw' dan 'proc'."""
    out = {}
    s = samples()
    env = npz(PACK / "audio" / "envelopes.npz")
    row = None
    if s is not None:
        r = s[s.file_id == fid]
        row = r.iloc[0] if len(r) else None
    for kind in ["raw", "proc"]:
        if row is not None and env is not None and isinstance(row.get(kind), str) and f"{fid}__{kind}" in env:
            out[kind] = (env[f"{fid}__{kind}"].astype(np.float32), int(row[f"{kind}_sr"]), float(row[f"{kind}_sec"]),
                         float(row[f"{kind}_peak"]), "/pack/" + row[kind])
            continue
        p = local_wav(machine, fid, kind)
        if p is not None:
            e, sr, dur, peak = _wav_info(str(p), _mtime(p))
            out[kind] = (e, sr, dur, peak, "/rawdata/" + p.relative_to(DATA).as_posix())
    return out


def local_wav(machine, fid, kind):
    if kind == "proc":
        p = PROCESSED / machine / f"{fid}.wav"
        return p if p.exists() else None
    pm, man = csv(PROC_MANIFEST), csv(MANIFEST)
    if pm is None or man is None:
        return None
    r = pm[pm.file_id == fid]
    raw_id = r.parent_id.iloc[0] if len(r) and "parent_id" in r and pd.notna(r.parent_id.iloc[0]) else fid
    m = man[man.file_id == raw_id]
    if not len(m):
        return None
    p = ROOT / m.path.iloc[0]
    return p if p.exists() else None


def raw_audio_url(file_id):
    """URL audio untuk halaman Dataset: klip paket (online) atau WAV lokal."""
    s = samples()
    if s is not None and "raw" in s:
        pm = csv(PROC_MANIFEST)
        kids = pm[pm.get("parent_id", pm.file_id) == file_id].file_id if pm is not None else []
        hit = s[s.file_id.isin(set(kids) | {file_id}) & s.raw.notna()]
        if len(hit):
            return "/pack/" + hit.raw.iloc[0]
    man = csv(MANIFEST)
    if man is not None:
        m = man[man.file_id == file_id]
        if len(m) and (ROOT / m.path.iloc[0]).exists():
            return "/rawdata/" + (ROOT / m.path.iloc[0]).relative_to(DATA).as_posix()
    return None


def url_to_path(url):
    """'/pack/...' atau '/rawdata/...' -> path file di disk (untuk st.audio)."""
    if not url:
        return None
    if url.startswith("/pack/"):
        p = PACK / url[len("/pack/"):]
    elif url.startswith("/rawdata/"):
        p = DATA / url[len("/rawdata/"):]
    else:
        return None
    return p if p.exists() else None


def packed_raw_ids():
    s = samples()
    pm = csv(PROC_MANIFEST)
    if s is None or "raw" not in s or pm is None:
        return []
    kids = s[s.raw.notna()].file_id
    par = pm.set_index("file_id").reindex(kids)
    col = par["parent_id"] if "parent_id" in par else pd.Series(kids.values, index=kids.values)
    return sorted(set(col.fillna(pd.Series(kids.values, index=kids.values))))


# ------------------------------------------------------------------ prediksi (tanpa numba)
def pred_available():
    avail = {}
    for f in sorted(PRED.glob("refs_*.npz")):
        task, feat = f.stem[len("refs_"):].rsplit("_", 1)
        avail.setdefault(task, {})[feat] = f
    return avail


@functools.lru_cache(maxsize=8)
def _refs(path, mtime):
    with np.load(path, allow_pickle=False) as z:
        Xr, Xq = z["Xr"].astype(np.float64), z["Xq"].astype(np.float64)
        mu = Xr.mean(axis=(0, 1), keepdims=True)[0]
        sd = Xr.std(axis=(0, 1), keepdims=True)[0] + 1e-8
        return dict(Xr=np.ascontiguousarray((Xr - mu) / sd), yr=z["yr"], idr=z["idr"],
                    Xq=np.ascontiguousarray((Xq - mu) / sd), yq=z["yq"], idq=z["idq"], mu=mu, sd=sd,
                    k=int(z["k"]), radius=int(z["radius"]), feature=str(z["feature"]), T=Xr.shape[1])


def refs(path):
    """Sama persis dengan _predict_lib.load_refs (z-score dari data referensi)."""
    return _refs(str(path), _mtime(path))


def pred_audio(fid):
    """Audio (processed) satu rekaman untuk halaman Prediksi: klip paket (online) atau WAV lokal. None bila tidak ada."""
    p = PACK / "audio" / "pred" / f"{fid}.ogg"
    if p.exists():
        return p
    s = samples()
    if s is not None and "proc" in s:
        r = s[(s.file_id == fid) & s.proc.notna()]
        if len(r) and (PACK / r.proc.iloc[0]).exists():
            return PACK / r.proc.iloc[0]
    pm = csv(PROC_MANIFEST)
    if pm is not None:
        r = pm[pm.file_id == fid]
        if len(r):
            p = PROCESSED / r.machine.iloc[0] / f"{fid}.wav"
            if p.exists():
                return p
    return None
