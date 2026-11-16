"""Tahap 8 - Paket data ringan untuk dashboard (Streamlit) -> folder dashboard_data/.

Tujuannya: dashboard TIDAK menghitung apa pun yang berat dan TIDAK membaca ribuan file.
Semua yang berat dihitung sekali di sini:
  * groups/<mesin>_secXX_<fitur>.npz : matriks DTW & FastDTW (float32), id rekaman, koordinat MDS (DTW & FastDTW)
  * features_<mesin>.npz            : pratinjau MFCC & Log-Mel untuk sejumlah contoh (float16, frame di-downsample)
  * audio/<mesin>/<id>_{raw,proc}.ogg : klip contoh untuk DIDENGAR saja (Ogg Vorbis, kecil) + envelope waveform
  * samples.csv                     : daftar contoh yang dipaketkan

Catatan: klip .ogg hanya untuk diputar di dashboard. Analisis tetap memakai WAV/fitur asli (lossless).

Contoh:  python 14_pack_for_dashboard.py                    (12 contoh per mesin)
         python 14_pack_for_dashboard.py --samples 30 --no-audio
Ukuran khas: beberapa MB saja -> aman diunggah ke GitHub / Streamlit Cloud."""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
from common import (CLUSTERING, DISTANCES, FEATURES, FEATURE_TYPES, MACHINES, MANIFEST, PROC_MANIFEST, PROCESSED,
                    ROOT, SR)

OUT = ROOT / "dashboard_data"
ENV_BINS = 1200          # resolusi envelope waveform (titik)
FEAT_MAX_FRAMES = 160    # pratinjau fitur: maksimal 160 frame (cukup untuk heatmap)


def mds_xy(D, seed=0):
    from sklearn.manifold import MDS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = MDS(2, dissimilarity="precomputed", random_state=seed, n_init=1, init="random")
        except TypeError:  # sklearn >= 1.8
            m = MDS(2, metric="precomputed", random_state=seed, n_init=1, init="random")
        return m.fit_transform(D).astype(np.float32)


def envelope(y, bins=ENV_BINS):
    """Envelope min/max per bin -> waveform tampak sama seperti aslinya, tapi hanya 2 x bins angka."""
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n == 0:
        return np.zeros((2, 1), np.float16)
    edges = np.linspace(0, n, min(bins, n) + 1).astype(int)
    lo = np.minimum.reduceat(y, edges[:-1])
    hi = np.maximum.reduceat(y, edges[:-1])
    return np.stack([lo, hi]).astype(np.float16)


def shrink_frames(A, T=FEAT_MAX_FRAMES):
    """A: (D, T0) -> (D, <=T) dengan rata-rata per blok frame."""
    D, T0 = A.shape
    if T0 <= T:
        return A
    edges = np.linspace(0, T0, T + 1).astype(int)
    return np.add.reduceat(A, edges[:-1], axis=1) / np.diff(edges)[None, :]


def write_ogg(path, y, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(path, y, sr, format="OGG", subtype="VORBIS")
        return path
    except Exception:  # libsndfile tanpa Vorbis -> FLAC (lossless, tetap ~2x lebih kecil dari WAV)
        p = path.with_suffix(".flac")
        sf.write(p, y, sr, format="FLAC")
        return p


def pick_samples(pm, n, seed):
    """Contoh stratified (mesin x domain x kondisi), diutamakan rekaman yang ikut dihitung jaraknya."""
    in_groups = set()
    for p in DISTANCES.glob("DTW/*/*_ids.csv"):
        in_groups |= set(pd.read_csv(p).file_id)
    out = []
    for machine, g in pm.groupby("machine"):
        g = g.assign(_pri=g.file_id.isin(in_groups)).sort_values(["_pri", "file_id"], ascending=[False, True])
        strata = list(g.groupby(["domain", "condition"]))
        per = max(1, int(np.ceil(n / max(1, len(strata)))))
        part = pd.concat([s.head(per * 3).sample(n=min(per, len(s)), random_state=seed) for _, s in strata])
        out.append(part.head(n))
    return pd.concat(out).drop(columns="_pri").sort_values(["machine", "file_id"]).reset_index(drop=True)


def pack_groups():
    (OUT / "groups").mkdir(parents=True, exist_ok=True)
    n = 0
    for machine in MACHINES:
        for f in sorted((DISTANCES / "DTW" / machine).glob("*_sec??.npy")):
            feat = f.stem.split("_sec")[0]
            if feat not in FEATURE_TYPES:
                continue
            ids = np.asarray(pd.read_csv(f.with_name(f.stem + "_ids.csv")).file_id.astype(str).tolist(), dtype=str)
            arrs = {"ids": ids}
            for dist in ["DTW", "FastDTW"]:
                p = DISTANCES / dist / machine / f.name
                if p.exists():
                    D = np.load(p)
                    arrs[dist] = D.astype(np.float32)
                    arrs[f"mds_{dist}"] = mds_xy(D)
            np.savez_compressed(OUT / "groups" / f"{machine}_{f.stem}.npz", **arrs)
            n += 1
            print(f"[grup] {machine} {f.stem}: N={len(ids)} -> {', '.join(k for k in arrs if k != 'ids')}")
    return n


def pack_features(samples):
    for machine, g in samples.groupby("machine"):
        arrs = {"ids": np.asarray(g.file_id.astype(str).tolist(), dtype=str)}
        for feat in FEATURE_TYPES:
            mats, shapes = [], []
            for fid in g.file_id:
                p = FEATURES / feat / machine / f"{fid}.npy"
                if p.exists():
                    A = np.load(p)
                    shapes.append(A.shape)
                    mats.append(shrink_frames(A).astype(np.float16))
                else:
                    shapes.append((0, 0))
                    mats.append(None)
            if all(m is None for m in mats):
                continue
            D = max(m.shape[0] for m in mats if m is not None)
            T = max(m.shape[1] for m in mats if m is not None)
            stack = np.full((len(mats), D, T), np.nan, np.float16)
            for i, m in enumerate(mats):
                if m is not None:
                    stack[i, :m.shape[0], :m.shape[1]] = m
            arrs[feat], arrs[f"{feat}_shape"] = stack, np.array(shapes)
        np.savez_compressed(OUT / f"features_{machine}.npz", **arrs)
        print(f"[fitur] {machine}: {len(g)} contoh")


def pack_audio(samples, man):
    env = {}
    rows = []
    for r in samples.itertuples():
        raw_id = getattr(r, "parent_id", r.file_id)
        raw_id = r.file_id if pd.isna(raw_id) else raw_id
        rec = dict(file_id=r.file_id, raw=None, proc=None)
        raw_p = ROOT / man.loc[raw_id, "path"] if raw_id in man.index else None
        proc_p = PROCESSED / r.machine / f"{r.file_id}.wav"
        for kind, p in [("raw", raw_p), ("proc", proc_p)]:
            if p is None or not p.exists():
                continue
            y, sr = sf.read(p, always_2d=True)
            y = y.mean(1).astype(np.float32)
            out = write_ogg(OUT / "audio" / r.machine / f"{r.file_id}_{kind}.ogg", y, sr)
            rec[kind] = str(out.relative_to(OUT)).replace("\\", "/")
            rec[f"{kind}_sr"], rec[f"{kind}_sec"], rec[f"{kind}_peak"] = sr, len(y) / sr, float(np.abs(y).max())
            env[f"{r.file_id}__{kind}"] = envelope(y)
        rows.append(rec)
    if env:
        np.savez_compressed(OUT / "audio" / "envelopes.npz", **env)
    return pd.DataFrame(rows)


def pack_pred_audio(pm, max_q=40, sr_out=8000):
    """Klip untuk halaman Prediksi: rekaman uji (maks. max_q per tugas) + rekaman referensi terdekatnya per kelas.
    Disimpan 8 kHz Ogg (hanya untuk didengar) supaya kecil."""
    from common import PROC_MANIFEST as _pmpath  # noqa: F401
    pred = ROOT / "prediction"
    files = sorted(pred.glob("refs_*.npz"))
    if not files:
        return 0
    import _predict_lib as P
    mach = pm.set_index("file_id").machine
    ids = set()
    for f in files:
        R = P.load_refs(f)
        q = np.arange(min(max_q, len(R["idq"])))
        ids |= set(map(str, R["idq"][q]))
        D = P.dtw_cross(np.ascontiguousarray(R["Xq"][q]), R["Xr"])
        for c in np.unique(R["yr"]):
            cols = np.where(R["yr"] == c)[0]
            ids |= set(map(str, R["idr"][cols[D[:, cols].argmin(1)]]))
        task, feat = f.stem[len("refs_"):].rsplit("_", 1)
        pc = ROOT / "results" / "prediction" / f"predictions_{task}_{feat}.csv"
        if pc.exists():
            pr = pd.read_csv(pc)
            pr = pr[pr.file_id.astype(str).isin(ids)]
            for col in ["nn_DTW", "nn_FastDTW", "nn_Euclid"]:
                if col in pr:
                    ids |= set(pr[col].astype(str))
    out = OUT / "audio" / "pred"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fid in sorted(ids):
        src = PROCESSED / str(mach.get(fid, "")) / f"{fid}.wav"
        dst = out / f"{fid}.ogg"
        if not src.exists() or dst.exists():
            n += dst.exists()
            continue
        y, sr = sf.read(src, always_2d=True)
        y = y.mean(1).astype(np.float32)
        if sr != sr_out:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_out)
        write_ogg(dst, y, sr_out)
        n += 1
    print(f"[prediksi] {n} klip audio untuk halaman Prediksi")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12, help="jumlah contoh audio/fitur per mesin")
    ap.add_argument("--no-audio", action="store_true", help="lewati pembuatan klip .ogg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred-audio", type=int, default=40, help="maks. rekaman uji per tugas yang audionya dipaketkan (0 = lewati)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    pm = pd.read_csv(PROC_MANIFEST)
    man = pd.read_csv(MANIFEST).set_index("file_id")

    n_groups = pack_groups()
    samples = pick_samples(pm, a.samples, a.seed)
    pack_features(samples)
    if not a.no_audio:
        au = pack_audio(samples, man)
        samples = samples.merge(au, on="file_id", how="left")
    samples.to_csv(OUT / "samples.csv", index=False)
    if not a.no_audio and a.pred_audio > 0:
        pack_pred_audio(pm, a.pred_audio)
    json.dump(dict(sr=SR, groups=n_groups, samples=len(samples), env_bins=ENV_BINS, feat_max_frames=FEAT_MAX_FRAMES),
              open(OUT / "info.json", "w"), indent=1)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"selesai -> {OUT}  ({size:.1f} MB)")
