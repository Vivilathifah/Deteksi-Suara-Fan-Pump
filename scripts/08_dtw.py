"""Tahap 4a - Matriks dissimilarity DTW exact (numba) per (mesin, section, fitur).
Karena DTW O(N^2 T^2), pakai --max-n untuk membatasi jumlah rekaman per grup (sampel stratified domain x kondisi)."""
import argparse, json, time
import numba
import numpy as np
import pandas as pd
from common import PROC_MANIFEST, DISTANCES, FEATURE_TYPES, MACHINES, ensure_dirs, list_groups, load_features, select_group
from _dtw_lib import dtw_matrix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--machines", nargs="+", default=MACHINES)
    ap.add_argument("--features", nargs="+", default=FEATURE_TYPES)
    ap.add_argument("--sections", nargs="+", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=numba.config.NUMBA_NUM_THREADS)
    a = ap.parse_args()
    ensure_dirs()
    numba.set_num_threads(a.n_jobs)
    pm = pd.read_csv(PROC_MANIFEST)
    for machine, sec in list_groups(pm):
        if machine not in a.machines or (a.sections and sec not in a.sections):
            continue
        g = select_group(pm, machine, sec, a.max_n, a.seed)
        for feat in a.features:
            X = load_features(feat, machine, g.file_id)
            dtw_matrix(X[:2])  # warm-up JIT agar tidak ikut waktu
            t0 = time.perf_counter()
            M = dtw_matrix(X)
            dt = time.perf_counter() - t0
            out = DISTANCES / "DTW" / machine
            np.save(out / f"{feat}_sec{sec:02d}.npy", M)
            g[["file_id"]].to_csv(out / f"{feat}_sec{sec:02d}_ids.csv", index=False)
            json.dump(dict(n=len(g), seconds=dt, n_jobs=a.n_jobs, feature=feat, machine=machine, section=sec),
                      open(out / f"{feat}_sec{sec:02d}_meta.json", "w"), indent=1)
            print(f"[DTW] {machine} sec{sec:02d} {feat}: N={len(g)} {dt:.1f}s")
