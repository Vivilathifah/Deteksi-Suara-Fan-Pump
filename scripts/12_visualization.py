"""Tahap 7 - Plot hasil ke results/plots (PNG)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from common import DISTANCES, CLUSTERING, RESULTS, PROC_MANIFEST, ensure_dirs, tag

P = RESULTS / "plots"

if __name__ == "__main__":
    ensure_dirs()
    cl_p, ap_p = RESULTS / "metrics" / "cluster_metrics.csv", RESULTS / "metrics" / "approximation_metrics.csv"
    cl = pd.read_csv(cl_p) if cl_p.exists() else pd.DataFrame()
    apx = pd.read_csv(ap_p) if ap_p.exists() else pd.DataFrame()

    # 1) metrik vs jumlah cluster
    if len(cl):
        for metric in ["silhouette", "PC", "PE"]:
            g = cl.groupby(["machine", "feature", "distance", "c"])[metric].mean().reset_index()
            fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, mc in zip(axs, ["fan", "pump"]):
                for (ft, ds), s in g[g.machine == mc].groupby(["feature", "distance"]):
                    ax.plot(s.c, s[metric], marker="o", label=f"{ft}-{ds}", ls="-" if ds == "DTW" else "--")
                ax.set_title(mc); ax.set_xlabel("c"); ax.grid(alpha=.3)
            axs[0].set_ylabel(metric); axs[1].legend(fontsize=7)
            fig.tight_layout(); fig.savefig(P / f"metric_{metric}_vs_c.png", dpi=150); plt.close(fig)

    # 2) scatter DTW vs FastDTW + bar waktu
    if len(apx):
        for r in apx.itertuples():
            a = np.load(DISTANCES / "DTW" / r.machine / f"{r.feature}_sec{r.section:02d}.npy")
            b = np.load(DISTANCES / "FastDTW" / r.machine / f"{r.feature}_sec{r.section:02d}.npy")
            iu = np.triu_indices_from(a, 1)
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(a[iu], b[iu], s=2, alpha=.3)
            lim = [min(a[iu].min(), b[iu].min()), max(a[iu].max(), b[iu].max())]
            ax.plot(lim, lim, "r--", lw=1); ax.set_xlabel("DTW"); ax.set_ylabel("FastDTW")
            ax.set_title(f"{r.machine} sec{r.section:02d} {r.feature}\nPearson={r.pearson:.3f}, MAPE={r.MAPE_pct:.2f}%")
            fig.tight_layout(); fig.savefig(P / f"scatter_dtw_fastdtw_{r.machine}_sec{r.section:02d}_{r.feature}.png", dpi=130); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        lab = apx.machine + "-s" + apx.section.astype(str) + "-" + apx.feature
        x = np.arange(len(apx)); ax.bar(x - .2, apx.pair_ms_dtw, .4, label="DTW"); ax.bar(x + .2, apx.pair_ms_fastdtw, .4, label="FastDTW")
        ax.set_xticks(x); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=7); ax.set_ylabel("ms / pasangan"); ax.legend()
        fig.tight_layout(); fig.savefig(P / "runtime_per_pair.png", dpi=150); plt.close(fig)

    # 3) proyeksi MDS + keanggotaan (c terbaik menurut silhouette)
    best_p = RESULTS / "tables" / "best_c_by_silhouette.csv"
    if best_p.exists():
        pm = pd.read_csv(PROC_MANIFEST).set_index("file_id")
        for r in pd.read_csv(best_p).itertuples():
            t = tag(r.machine, r.section, r.feature, r.distance)
            D = np.load(DISTANCES / r.distance / r.machine / f"{r.feature}_sec{r.section:02d}.npy")
            mem = pd.read_csv(CLUSTERING / "membership" / f"{t}_c{r.c}.csv")
            xy = MDS(2, dissimilarity="precomputed", random_state=0, n_init=1, init="random").fit_transform(D)
            U = mem.filter(like="u").values
            fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
            sc = axs[0].scatter(xy[:, 0], xy[:, 1], c=U.argmax(1), s=10 + 60 * (U.max(1) - U.min()), cmap="tab10", alpha=.7)
            axs[0].set_title(f"{t} c={r.c} (ukuran = keyakinan keanggotaan)")
            for k, (nm, col) in enumerate([("domain", "domain"), ("condition", "condition")][:1]):
                for v in sorted(pm.loc[mem.file_id, col].unique()):
                    mk = pm.loc[mem.file_id, col].values == v
                    axs[1].scatter(xy[mk, 0], xy[mk, 1], s=10, label=v, alpha=.7)
                axs[1].legend(); axs[1].set_title("warna = domain (source/target)")
            fig.tight_layout(); fig.savefig(P / f"mds_{t}_c{r.c}.png", dpi=130); plt.close(fig)
    print("plot tersimpan di", P)
