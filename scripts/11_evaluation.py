"""Tahap 6 - Evaluasi: (a) kualitas cluster, (b) kecepatan, (c) ketepatan aproksimasi FastDTW terhadap DTW."""
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi, silhouette_score
from common import DISTANCES, CLUSTERING, RESULTS, PROC_MANIFEST, DISTANCE_TYPES, FEATURE_TYPES, MACHINES, ensure_dirs, tag


def fuzzy_indices(U):
    N, c = U.shape
    pc = (U ** 2).sum() / N
    pe = -(U * np.log(np.clip(U, 1e-12, None))).sum() / N
    return pc, (c * pc - 1) / (c - 1), pe  # PC, MPC, PE


if __name__ == "__main__":
    ensure_dirs()
    pm = pd.read_csv(PROC_MANIFEST).set_index("file_id")
    cl_rows, ap_rows = [], []
    for machine in MACHINES:
        for f in sorted((DISTANCES / "DTW" / machine).glob("*_sec??.npy")):
            feat, sec = f.stem.split("_sec")
            sec = int(sec)
            if feat not in FEATURE_TYPES:
                continue
            Ds, ids, meta = {}, None, {}
            for dist in DISTANCE_TYPES:
                p = DISTANCES / dist / machine / f.name
                if p.exists():
                    Ds[dist] = np.load(p)
                    meta[dist] = json.load(open(p.with_name(p.stem + "_meta.json")))
                    ids = pd.read_csv(p.with_name(p.stem + "_ids.csv")).file_id
            if ids is None:
                continue
            dom, cond = pm.loc[ids, "domain"].values, pm.loc[ids, "condition"].values
            # (a) kualitas cluster
            for dist, D in Ds.items():
                t = tag(machine, sec, feat, dist)
                lab_p = CLUSTERING / "labels" / f"{t}.csv"
                if not lab_p.exists():
                    continue
                labs = pd.read_csv(lab_p)
                for col in [c for c in labs.columns if c.startswith("label_c")]:
                    c = int(col.split("c")[-1])
                    U = pd.read_csv(CLUSTERING / "membership" / f"{t}_c{c}.csv").filter(like="u").values
                    y = labs[col].values
                    pc, mpc, pe = fuzzy_indices(U)
                    sil = silhouette_score(D, y, metric="precomputed") if len(set(y)) > 1 else np.nan
                    cl_rows.append(dict(machine=machine, section=sec, feature=feat, distance=dist, c=c, n=len(y), PC=pc, MPC=mpc, PE=pe,
                                        silhouette=sil, ARI_domain=ari(dom, y), NMI_domain=nmi(dom, y),
                                        ARI_condition=ari(cond, y), NMI_condition=nmi(cond, y)))
            # (b)+(c) aproksimasi FastDTW
            if len(Ds) == 2:
                iu = np.triu_indices_from(Ds["DTW"], 1)
                a, b = Ds["DTW"][iu], Ds["FastDTW"][iu]
                rel = (b - a) / np.maximum(a, 1e-12)
                row = dict(machine=machine, section=sec, feature=feat, n=len(ids), radius=meta["FastDTW"].get("radius"),
                           pearson=pearsonr(a, b)[0], spearman=spearmanr(a, b)[0], MAE=np.abs(b - a).mean(),
                           MAPE_pct=np.abs(rel).mean() * 100, max_rel_err_pct=np.abs(rel).max() * 100, mean_signed_rel_pct=rel.mean() * 100,
                           sec_dtw=meta["DTW"]["seconds"], sec_fastdtw=meta["FastDTW"]["seconds"],
                           pair_ms_dtw=meta["FastDTW"]["pair_ms_dtw"], pair_ms_fastdtw=meta["FastDTW"]["pair_ms_fastdtw"],
                           pair_ms_dtw_python=meta["FastDTW"].get("pair_ms_dtw_python"))
                row["speedup_pair_(DTW/FastDTW)"] = row["pair_ms_dtw"] / row["pair_ms_fastdtw"]
                row["speedup_vs_python_dtw"] = (row["pair_ms_dtw_python"] / row["pair_ms_fastdtw"]) if row["pair_ms_dtw_python"] else None
                ap_rows.append(row)
    cl = pd.DataFrame(cl_rows)
    apx = pd.DataFrame(ap_rows)
    # konsistensi cluster DTW vs FastDTW
    if len(cl):
        agree = []
        for (m, s, ft), _ in cl.groupby(["machine", "section", "feature"]):
            pa, pb = CLUSTERING / "labels" / f"{tag(m, s, ft, 'DTW')}.csv", CLUSTERING / "labels" / f"{tag(m, s, ft, 'FastDTW')}.csv"
            if pa.exists() and pb.exists():
                A, B = pd.read_csv(pa), pd.read_csv(pb)
                for col in [c for c in A.columns if c.startswith("label_c")]:
                    agree.append(dict(machine=m, section=s, feature=ft, c=int(col.split("c")[-1]), ARI_DTW_vs_FastDTW=ari(A[col], B[col])))
        agree = pd.DataFrame(agree)
        agree.to_csv(RESULTS / "metrics" / "cluster_agreement.csv", index=False)
    cl.to_csv(RESULTS / "metrics" / "cluster_metrics.csv", index=False)
    apx.to_csv(RESULTS / "metrics" / "approximation_metrics.csv", index=False)
    if len(cl):
        best = cl.loc[cl.groupby(["machine", "section", "feature", "distance"]).silhouette.idxmax()]
        best.to_csv(RESULTS / "tables" / "best_c_by_silhouette.csv", index=False)
        cl.groupby(["machine", "feature", "distance"])[["PC", "MPC", "PE", "silhouette", "ARI_domain", "ARI_condition"]].mean().round(4)\
            .to_csv(RESULTS / "tables" / "summary_by_machine_feature_distance.csv")
    if len(apx):
        apx.groupby(["machine", "feature"])[["pearson", "spearman", "MAPE_pct", "speedup_pair_(DTW/FastDTW)", "speedup_vs_python_dtw"]].mean().round(4)\
            .to_csv(RESULTS / "tables" / "summary_fastdtw_approximation.csv")
    print(cl.round(3).head(20).to_string() if len(cl) else "belum ada hasil clustering")
    print(apx.round(3).to_string() if len(apx) else "belum ada pasangan DTW/FastDTW")
