"""Tahap 1d / 2a - Validasi kualitas audio: terbaca, NaN/Inf, clipping, silence, durasi, sampling rate."""
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from common import MANIFEST, ROOT, RESULTS, ensure_dirs

if __name__ == "__main__":
    ensure_dirs()
    m = pd.read_csv(MANIFEST)
    rows = []
    for r in tqdm(m.itertuples(), total=len(m), desc="validasi"):
        issue = []
        try:
            y, sr = sf.read(ROOT / r.path, always_2d=True)
            y = y.mean(axis=1)
            peak, rms = float(np.abs(y).max()), float(np.sqrt((y ** 2).mean()))
            clip = float((np.abs(y) >= 0.999).mean())
            if not np.isfinite(y).all(): issue.append("nan_inf")
            if clip > 0.001: issue.append("clipping")
            if rms < 1e-4: issue.append("silent")
            if abs(len(y) / sr - 10.0) > 0.1: issue.append("duration!=10s")
            if sr != 16000: issue.append(f"sr={sr}")
        except Exception as e:
            peak = rms = clip = np.nan
            issue.append(f"unreadable:{type(e).__name__}")
        rows.append(dict(file_id=r.file_id, peak=peak, rms=rms, clip_ratio=clip, issues=";".join(issue), valid=not any(
            i in ("nan_inf", "silent") or i.startswith("unreadable") for i in issue)))
    v = pd.DataFrame(rows)
    v.to_csv(RESULTS / "tables" / "audio_validation.csv", index=False)
    print(f"valid: {int(v.valid.sum())}/{len(v)}")
    print(v[v.issues != ""].issues.value_counts().to_string())
