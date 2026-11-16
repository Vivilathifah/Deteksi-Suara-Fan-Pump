"""Tahap 2 - Preprocessing: resample 16 kHz, mono, normalisasi puncak, segmentasi opsional.
Contoh: python 05_preprocessing.py                  (tanpa segmentasi, 1 file = 1 rekaman 10 s)
        python 05_preprocessing.py --segment-sec 2  (potong 2 s tanpa overlap)"""
import argparse
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm
from common import MANIFEST, PROC_MANIFEST, PROCESSED, ROOT, RESULTS, SR, ensure_dirs


def process(r, seg_sec):
    y, _ = librosa.load(ROOT / r["path"], sr=SR, mono=True)
    y = y - y.mean()
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.99
    out_dir = PROCESSED / r["machine"]
    segs = [y]
    if seg_sec:
        L = int(seg_sec * SR)
        segs = [y[i:i + L] for i in range(0, len(y) - L + 1, L)]
    rows = []
    for k, s in enumerate(segs):
        fid = r["file_id"] if not seg_sec else f"{r['file_id']}_seg{k:02d}"
        sf.write(out_dir / f"{fid}.wav", s, SR)
        rows.append({**r, "file_id": fid, "parent_id": r["file_id"], "segment": k if seg_sec else -1,
                     "n_samples": len(s), "sr": SR})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment-sec", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    ensure_dirs()
    m = pd.read_csv(MANIFEST)
    vp = RESULTS / "tables" / "audio_validation.csv"
    if vp.exists():
        ok = pd.read_csv(vp).query("valid").file_id
        m = m[m.file_id.isin(ok)]
    recs = m.to_dict("records")
    res = Parallel(n_jobs=a.n_jobs)(delayed(process)(r, a.segment_sec) for r in tqdm(recs, desc="preprocess"))
    pm = pd.DataFrame([x for rr in res for x in rr])
    pm.to_csv(PROC_MANIFEST, index=False)
    print(f"{len(pm)} file hasil -> {PROC_MANIFEST}")
