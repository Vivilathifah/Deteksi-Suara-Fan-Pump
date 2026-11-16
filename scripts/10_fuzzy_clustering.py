"""Tahap 5 - Fuzzy C-Medoids (relational, Krishnapuram et al. 2001) pada matriks jarak DTW/FastDTW.
Output: derajat keanggotaan (clustering/membership), label hard argmax (clustering/labels), model medoid (clustering/models)."""
import argparse, json
import numpy as np
import pandas as pd
from common import DISTANCES, CLUSTERING, DISTANCE_TYPES, FEATURE_TYPES, MACHINES, ensure_dirs, tag


def fcmdd(D, c, m=2.0, n_init=10, max_iter=100, seed=0):
    N = len(D)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        # inisialisasi medoid ala k-means++ pada matriks jarak
        med = [int(rng.integers(N))]
        while len(med) < c:
            d2 = D[:, med].min(axis=1) ** 2
            med.append(int(rng.choice(N, p=d2 / d2.sum())) if d2.sum() > 0 else int(rng.integers(N)))
        med = np.array(med)
        for it in range(max_iter):
            d = np.maximum(D[:, med], 1e-12)
            inv = d ** (-1.0 / (m - 1))
            U = inv / inv.sum(axis=1, keepdims=True)
            Um = U ** m
            new = (Um.T @ D).argmin(axis=1)
            if len(set(new)) < c or set(new) == set(med):
                break
            med = new
        d = np.maximum(D[:, med], 1e-12)
        inv = d ** (-1.0 / (m - 1))
        U = inv / inv.sum(axis=1, keepdims=True)
        J = float(((U ** m) * D[:, med]).sum())
        if best is None or J < best[0]:
            best = (J, med.copy(), U, it + 1)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-min", type=int, default=2)
    ap.add_argument("--c-max", type=int, default=6)
    ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ensure_dirs()
    for dist in DISTANCE_TYPES:
        for machine in MACHINES:
            for f in sorted((DISTANCES / dist / machine).glob("*_sec??.npy")):
                feat, sec = f.stem.split("_sec")
                sec = int(sec)
                if feat not in FEATURE_TYPES:
                    continue
                D = np.load(f)
                ids = pd.read_csv(f.with_name(f.stem + "_ids.csv")).file_id
                t = tag(machine, sec, feat, dist)
                labels = pd.DataFrame({"file_id": ids})
                models = {}
                for c in range(a.c_min, a.c_max + 1):
                    J, med, U, it = fcmdd(D, c, a.m, a.n_init, seed=a.seed)
                    labels[f"label_c{c}"] = U.argmax(axis=1)
                    mem = pd.DataFrame(U, columns=[f"u{k+1}" for k in range(c)])
                    mem.insert(0, "file_id", ids)
                    mem.to_csv(CLUSTERING / "membership" / f"{t}_c{c}.csv", index=False)
                    models[c] = dict(medoid_idx=[int(x) for x in med], medoid_ids=[ids[i] for i in med], objective=J, iterations=it, m=a.m)
                labels.to_csv(CLUSTERING / "labels" / f"{t}.csv", index=False)
                json.dump(models, open(CLUSTERING / "models" / f"{t}.json", "w"), indent=1)
                print(f"[FCMdd] {t}: c={a.c_min}..{a.c_max}")
