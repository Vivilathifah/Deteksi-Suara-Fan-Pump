"""Tahap 7 - Prediksi k-NN (1-NN) dengan jarak DTW, FastDTW, dan Euclid (baseline).
Alur (mengikuti materi kuliah): fitur MFCC -> jarak ke rekaman referensi (template) -> label tetangga terdekat = prediksi.

Tugas:
  condition_fan / condition_pump : normal vs anomali (per mesin, hanya split 'test' karena anomali hanya ada di sana)
  machine                        : jenis mesin (fan vs pump)

Contoh:  python 13_predict.py --max-n 100 --features MFCC
Keluaran: results/prediction/{metrics.csv, predictions_<tugas>_<fitur>.csv}, prediction/refs_<tugas>_<fitur>.npz (untuk dashboard)."""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from common import FEATURES, PROC_MANIFEST, ensure_dirs
from _predict_lib import (PRED, PRED_RES, METHODS, dtw_cross, euclid_cross, fastdtw_cross, fit_scaler,
                          knn_predict, save_refs, warmup)
import time

TASKS = {"condition_fan": "condition", "condition_pump": "condition", "machine": "machine"}


def build_pool(pm, task, sections):
    if task.startswith("condition_"):
        df = pm[(pm.machine == task.split("_")[1]) & (pm.split == "test")]
    else:
        df = pm
    if sections:
        df = df[df.section.isin(sections)]
    df = df.copy()
    df["_parent"] = df["parent_id"] if "parent_id" in df.columns else df["file_id"]
    return df


def sample_parents(par, label, max_n, seed):
    """Kelas seimbang: maks max_n/jumlah_kelas rekaman per kelas, stratified menurut domain."""
    classes = sorted(par[label].unique())
    per = max(1, max_n // len(classes))
    parts = []
    for c in classes:
        g = par[par[label] == c]
        if len(g) <= per:
            parts.append(g)
            continue
        frac = per / len(g)
        for _, s in g.groupby("domain"):
            parts.append(s.sample(n=max(1, round(len(s) * frac)), random_state=seed))
    return pd.concat(parts)


def split_parents(par, label, test_frac, seed):
    te = []
    for _, g in par.groupby(label):
        n_te = min(max(1, round(len(g) * test_frac)), len(g) - 1)
        if n_te > 0:
            te.append(g.sample(n=n_te, random_state=seed))
    te = pd.concat(te)
    return par.drop(te.index), te


def load_X(rows, feat):
    return np.stack([np.load(FEATURES / feat / r.machine / f"{r.file_id}.npy").T for r in rows.itertuples()]).astype(np.float64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--features", nargs="+", default=["MFCC"])
    ap.add_argument("--max-n", type=int, default=100, help="maks rekaman (induk) per tugas, kelas seimbang")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--sections", nargs="+", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=2, help="worker FastDTW (jaga kecil agar RAM aman)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ensure_dirs()
    PRED.mkdir(exist_ok=True)
    PRED_RES.mkdir(parents=True, exist_ok=True)
    pm = pd.read_csv(PROC_MANIFEST)
    warmup()
    rows = []
    for task in a.tasks:
        label = TASKS[task]
        pool = build_pool(pm, task, a.sections)
        if pool[label].nunique() < 2:
            print(f"[{task}] dilewati: hanya 1 kelas pada data ({pool[label].unique()})")
            continue
        par = pool.drop_duplicates("_parent")
        par = sample_parents(par, label, a.max_n, a.seed)
        ref_par, te_par = split_parents(par, label, a.test_frac, a.seed)
        ref = pool[pool._parent.isin(ref_par._parent)].reset_index(drop=True)
        qry = pool[pool._parent.isin(te_par._parent)].reset_index(drop=True)
        yr, yq = ref[label].astype(str).values, qry[label].astype(str).values
        for feat in a.features:
            Xr, Xq = load_X(ref, feat), load_X(qry, feat)
            mu, sd = fit_scaler(Xr)
            Rs, Qs = (Xr - mu) / sd, (Xq - mu) / sd
            D, secs = {}, {}
            for name, fn in [("DTW", lambda: dtw_cross(Qs, Rs)), ("FastDTW", lambda: fastdtw_cross(Qs, Rs, a.radius, a.n_jobs)),
                             ("Euclid", lambda: euclid_cross(Qs, Rs))]:
                t0 = time.perf_counter()
                D[name] = fn()
                secs[name] = time.perf_counter() - t0
            pred = {m: knn_predict(D[m], yr, a.k) for m in METHODS}
            out = pd.DataFrame({"file_id": qry.file_id, "y_true": yq})
            for m in METHODS:
                nn = D[m].argmin(axis=1)
                out[f"pred_{m}"], out[f"nn_{m}"], out[f"d_{m}"] = pred[m], ref.file_id.values[nn], D[m].min(axis=1)
            out.to_csv(PRED_RES / f"predictions_{task}_{feat}.csv", index=False)
            for m in METHODS:
                kw = dict(average="macro", zero_division=0)
                rows.append(dict(task=task, feature=feat, method=m, k=a.k, radius=a.radius, n_ref=len(ref), n_test=len(qry),
                                 accuracy=accuracy_score(yq, pred[m]), f1_macro=f1_score(yq, pred[m], **kw),
                                 precision_macro=precision_score(yq, pred[m], **kw), recall_macro=recall_score(yq, pred[m], **kw),
                                 seconds_total=secs[m], ms_per_query=secs[m] / len(qry) * 1e3,
                                 agree_with_DTW=float((pred[m] == pred["DTW"]).mean())))
                print(f"[{task}/{feat}] {m:8s} acc={rows[-1]['accuracy']:.3f}  f1={rows[-1]['f1_macro']:.3f}  {rows[-1]['ms_per_query']:.1f} ms/query")
            save_refs(PRED / f"refs_{task}_{feat}.npz", Xr, yr, ref.file_id, Xq, yq, qry.file_id, a.k, a.radius, feat)
    new = pd.DataFrame(rows)
    mp = PRED_RES / "metrics.csv"
    if mp.exists() and len(new):
        old = pd.read_csv(mp)
        old = old[~old.set_index(["task", "feature"]).index.isin(new.set_index(["task", "feature"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    if len(new):
        new.to_csv(mp, index=False)
        print(f"metrik -> {mp}")
