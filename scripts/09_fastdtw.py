"""Tahap 4b - Matriks dissimilarity FastDTW (sampel identik dengan 08_dtw.py) + benchmark waktu per-pasangan."""
import argparse, json, time
import numpy as np
import pandas as pd
from common import PROC_MANIFEST, DISTANCES, FEATURE_TYPES, MACHINES, ensure_dirs, list_groups, load_features, select_group
from _dtw_lib import dtw_pair, fastdtw_pair, fastdtw_matrix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--machines", nargs="+", default=MACHINES)
    ap.add_argument("--features", nargs="+", default=FEATURE_TYPES)
    ap.add_argument("--sections", nargs="+", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--bench-pairs", type=int, default=100)
    ap.add_argument("--bench-python-pairs", type=int, default=2, help="pasangan utk DTW python murni (pembanding adil FastDTW); 0 = lewati")
    a = ap.parse_args()
    ensure_dirs()
    pm = pd.read_csv(PROC_MANIFEST)
    rng = np.random.default_rng(a.seed)
    for machine, sec in list_groups(pm):
        if machine not in a.machines or (a.sections and sec not in a.sections):
            continue
        g = select_group(pm, machine, sec, a.max_n, a.seed)
        for feat in a.features:
            X = load_features(feat, machine, g.file_id)
            N = len(X)
            # benchmark sekuensial per pasangan (perbandingan kecepatan yang adil, 1 core)
            pairs = [tuple(rng.choice(N, 2, replace=False)) for _ in range(min(a.bench_pairs, N * (N - 1) // 2))]
            dtw_pair(X[0], X[1])
            t0 = time.perf_counter(); [dtw_pair(X[i], X[j]) for i, j in pairs]; t_dtw = (time.perf_counter() - t0) / len(pairs)
            t0 = time.perf_counter(); [fastdtw_pair(X[i], X[j], a.radius) for i, j in pairs]; t_fast = (time.perf_counter() - t0) / len(pairs)
            t_py = np.nan
            if a.bench_python_pairs:
                t0 = time.perf_counter(); [dtw_pair.py_func(X[i], X[j]) for i, j in pairs[:a.bench_python_pairs]]
                t_py = (time.perf_counter() - t0) / a.bench_python_pairs
            t0 = time.perf_counter()
            M = fastdtw_matrix(X, a.radius, a.n_jobs)
            dt = time.perf_counter() - t0
            out = DISTANCES / "FastDTW" / machine
            np.save(out / f"{feat}_sec{sec:02d}.npy", M)
            g[["file_id"]].to_csv(out / f"{feat}_sec{sec:02d}_ids.csv", index=False)
            json.dump(dict(n=N, seconds=dt, n_jobs=a.n_jobs, radius=a.radius, feature=feat, machine=machine, section=sec,
                           pair_ms_dtw=t_dtw * 1e3, pair_ms_fastdtw=t_fast * 1e3, pair_ms_dtw_python=t_py * 1e3),
                      open(out / f"{feat}_sec{sec:02d}_meta.json", "w"), indent=1)
            print(f"[FastDTW] {machine} sec{sec:02d} {feat}: N={N} {dt:.1f}s | per-pair DTW {t_dtw*1e3:.2f} ms vs FastDTW {t_fast*1e3:.2f} ms | DTW python murni {t_py*1e3:.0f} ms")
